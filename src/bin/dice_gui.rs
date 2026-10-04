//! Simple egui GUI for the dice roller (`dice-gui` binary).
//!
//! The roll loop runs on a background thread via the shared parallel
//! engine in `dice::run_rolls_parallel`; the UI thread only renders
//! events it receives over an mpsc channel, so the interface stays
//! responsive at billions of rolls/sec.

use dice::{
    MAX_DICE, NO_CAP, expected_rolls, fmt_comma, LuckyConfig, LuckyEvent, LuckyOutcome,
    RollConfig, RollEvent, RollOutcome, find_lucky_seed, lucky_threshold, run_rolls_parallel,
};
use eframe::egui;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc,
};
use std::thread;
use std::time::Duration;

const DEFAULT_MAX_ROLLS: &str = "15000000000";
const MAX_LOG_LINES: usize = 300;

enum GuiMsg {
    Event(RollEvent),
    Done(RollOutcome),
    LuckyProgress {
        seeds_tried: u64,
        total_rolls: u64,
        best_rolls: Option<u64>,
        elapsed_secs: f64,
    },
    LuckyDone(LuckyOutcome),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Idle,
    Running,
    Won,
    MaxRolls,
    Stopped,
    Error,
    LuckyFound,
    Exhausted,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Idle => "Idle",
            Status::Running => "Rolling…",
            Status::Won => "Match found",
            Status::MaxRolls => "Stopped (max rolls)",
            Status::Stopped => "Stopped",
            Status::Error => "Error",
            Status::LuckyFound => "Lucky seed found",
            Status::Exhausted => "No lucky seed",
        }
    }
}

struct DiceApp {
    n_text: String,
    x_text: String,
    seed_text: String,
    max_text: String,
    threads_text: String,
    lucky_mode: bool,
    pct_text: String,
    max_seeds_text: String,
    replay_seed: Option<u64>,
    lucky_seeds: u64,
    lucky_rolls: u64,
    lucky_best: Option<u64>,
    error: String,
    expected_note: String,
    status: Status,
    rolls: u64,
    elapsed_secs: f64,
    last: Vec<u32>,
    log: Vec<String>,
    result: String,
    rx: Option<mpsc::Receiver<GuiMsg>>,
    stop: Option<Arc<AtomicBool>>,
}

impl Default for DiceApp {
    fn default() -> Self {
        Self {
            n_text: "2".to_owned(),
            x_text: "6".to_owned(),
            seed_text: String::new(),
            max_text: DEFAULT_MAX_ROLLS.to_owned(),
            threads_text: "0".to_owned(),
            lucky_mode: false,
            pct_text: "10".to_owned(),
            max_seeds_text: "100000".to_owned(),
            replay_seed: None,
            lucky_seeds: 0,
            lucky_rolls: 0,
            lucky_best: None,
            error: String::new(),
            expected_note: String::new(),
            status: Status::Idle,
            rolls: 0,
            elapsed_secs: 0.0,
            last: Vec::new(),
            log: Vec::new(),
            result: String::new(),
            rx: None,
            stop: None,
        }
    }
}

/// Parse a u64 that may contain commas, underscores or spaces (e.g. "15,000,000,000").
fn parse_u64_loose(s: &str) -> Option<u64> {
    let cleaned: String = s
        .chars()
        .filter(|c| *c != ',' && *c != '_' && *c != ' ' && *c != '\'')
        .collect();
    if cleaned.is_empty() {
        return None;
    }
    cleaned.parse().ok()
}

impl DiceApp {
    fn running(&self) -> bool {
        self.status == Status::Running
    }

    /// Shared dice/seed validation (n, x, optional seed). Used by normal
    /// and lucky modes; each mode adds its own remaining fields.
    /// The seed slot doubles as the lucky-search base (empty = 0).
    fn parse_dice(&self) -> Result<(usize, u32, Option<u64>), String> {
        let n: usize = self
            .n_text
            .trim()
            .parse()
            .map_err(|_| "n must be an integer >= 1".to_owned())?;
        let x: u32 = self
            .x_text
            .trim()
            .parse()
            .map_err(|_| "x must be an integer >= 2".to_owned())?;
        if n < 1 {
            return Err("n must be >= 1".to_owned());
        }
        if n > MAX_DICE {
            return Err(format!("n must be <= {} dice", fmt_comma(MAX_DICE as u128)));
        }
        if x < 2 {
            return Err("x must be >= 2".to_owned());
        }
        let seed: Option<u64> = if self.seed_text.trim().is_empty() {
            None
        } else {
            Some(
                self.seed_text
                    .trim()
                    .parse::<i64>()
                    .map_err(|_| "seed must be an integer (empty = random)".to_owned())?
                    as u64,
            )
        };
        Ok((n, x, seed))
    }

    /// Max-rolls cap, `None` (unlimited) when empty or `0`.
    fn parse_cap(&self) -> Result<Option<u64>, String> {
        // Empty or 0 = unlimited (None); otherwise a cap >= 1.
        let cap: Option<u64> = if self.max_text.trim().is_empty() {
            None
        } else {
            match parse_u64_loose(&self.max_text) {
                Some(0) => None,
                Some(v) => Some(v),
                None => {
                    return Err(
                        "max rolls must be a positive integer (empty = unlimited)".to_owned(),
                    );
                }
            }
        };
        Ok(cap)
    }

    /// Full normal-mode validation: dice plus the max-rolls cap.
    fn parse_inputs(&self) -> Result<(usize, u32, Option<u64>, Option<u64>), String> {
        let (n, x, seed) = self.parse_dice()?;
        Ok((n, x, seed, self.parse_cap()?))
    }

    /// Lucky-search fields: percent of expected rolls plus max seeds
    /// to try (empty = unlimited).
    fn parse_lucky(&self) -> Result<(f64, u64), String> {
        let pct: f64 = self
            .pct_text
            .trim()
            .parse()
            .map_err(|_| "luck % must be a positive number".to_owned())?;
        if !(pct.is_finite() && pct > 0.0) {
            return Err("luck % must be a positive number".to_owned());
        }
        let max_seeds: u64 = if self.max_seeds_text.trim().is_empty() {
            0
        } else {
            parse_u64_loose(&self.max_seeds_text).ok_or_else(|| {
                "max seeds must be a non-negative integer (empty = unlimited)".to_owned()
            })?
        };
        Ok((pct, max_seeds))
    }

    /// Parse the threads field, preserving 0 = automatic. Callers resolve
    /// it with [`resolve_threads`].
    fn parse_threads(&self) -> Result<usize, String> {
        self.threads_text
            .trim()
            .parse()
            .map_err(|_| "threads must be a non-negative integer (0 = auto)".to_owned())
    }

/// All CPUs, falling back to 1.
fn auto_threads() -> usize {
    std::thread::available_parallelism()
        .map(|nz| nz.get())
        .unwrap_or(1)
}

/// Resolve a requested thread count: explicit values win; auto (0) uses
/// all CPUs, except an explicit seed forces single-threaded for exact
/// reproduction (matching lucky-search evaluation, so replay matches).
/// Pass `None` for the seed where no such constraint exists (lucky
/// search workers: the result is thread-count independent anyway).
fn resolve_threads(requested: usize, seed: Option<u64>) -> usize {
    if requested != 0 {
        requested
    } else if seed.is_some() {
        1
    } else {
        Self::auto_threads()
    }
}

    fn expected_note_for(n: usize, x: u32, cap: Option<u64>) -> String {
        match cap {
            // No cap to exceed: plain estimate.
            None => format!(
                "Expected rolls ~{}.",
                fmt_comma(expected_rolls(n, x, NO_CAP).0)
            ),
            Some(m) => {
                let (expected, exceeds) = expected_rolls(n, x, m);
                if exceeds {
                    format!(
                        "Expected rolls ~{} exceed max rolls {}.",
                        fmt_comma(expected),
                        fmt_comma(m as u128)
                    )
                } else {
                    format!("Expected rolls ~{}.", fmt_comma(expected))
                }
            }
        }
    }

    fn start_run(&mut self) {
        self.error.clear();
        self.result.clear();
        self.expected_note.clear();
        self.replay_seed = None;

        let (n, x, seed) = match self.parse_dice() {
            Ok(v) => v,
            Err(msg) => {
                self.fail(&msg);
                return;
            }
        };
        let threads_req = match self.parse_threads() {
            Ok(t) => t,
            Err(msg) => {
                self.fail(&msg);
                return;
            }
        };

        if self.lucky_mode {
            // Search workers stay fully parallel: the found seed is
            // thread-count independent, so speed costs nothing here.
            self.start_lucky_run(n, x, seed, Self::resolve_threads(threads_req, None));
            return;
        }
        // An explicit seed pins single-threaded for exact reproduction
        // (matching lucky evaluation, so replay matches) unless Threads
        // is set explicitly.
        let threads = Self::resolve_threads(threads_req, seed);

        let cap = match self.parse_cap() {
            Ok(v) => v,
            Err(msg) => {
                self.fail(&msg);
                return;
            }
        };
        self.expected_note = Self::expected_note_for(n, x, cap);

        let (tx, rx) = mpsc::channel::<GuiMsg>();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let config = RollConfig {
            n,
            x,
            seed,
            max_rolls: cap.unwrap_or(NO_CAP),
            threads,
        };
        thread::spawn(move || {
            let outcome = run_rolls_parallel(&config, &worker_stop, |ev| {
                let _ = tx.send(GuiMsg::Event(ev));
            });
            let _ = tx.send(GuiMsg::Done(outcome));
        });

        self.rx = Some(rx);
        self.stop = Some(stop);
        self.status = Status::Running;
        self.rolls = 0;
        self.elapsed_secs = 0.0;
        self.last.clear();
        self.log.clear();
        let cap_note = match cap {
            None => "no cap".to_owned(),
            Some(m) => format!("max {}", fmt_comma(m as u128)),
        };
        if threads > 1 {
            self.log.push(format!(
                "Rolling {n}d{x} on {threads} threads ({cap_note})…"
            ));
        } else {
            self.log
                .push(format!("Rolling {n}d{x} ({cap_note})…"));
        }
    }

    fn start_lucky_run(&mut self, n: usize, x: u32, seed: Option<u64>, threads: usize) {
        let (pct, max_seeds) = match self.parse_lucky() {
            Ok(v) => v,
            Err(msg) => {
                self.fail(&msg);
                return;
            }
        };
        let threshold = lucky_threshold(n, x, pct);
        let base = seed.unwrap_or(0);
        self.expected_note = format!(
            "Need win within {} rolls ({pct}% of expected ~{}).",
            fmt_comma(threshold as u128),
            fmt_comma(expected_rolls(n, x, NO_CAP).0),
        );

        let (tx, rx) = mpsc::channel::<GuiMsg>();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let config = LuckyConfig {
            n,
            x,
            base_seed: base,
            threshold,
            max_seeds,
            threads,
        };
        thread::spawn(move || {
            let outcome = find_lucky_seed(&config, &worker_stop, |ev| {
                let LuckyEvent::Progress {
                    seeds_tried,
                    total_rolls,
                    best_rolls,
                    elapsed_secs,
                } = ev;
                let _ = tx.send(GuiMsg::LuckyProgress {
                    seeds_tried,
                    total_rolls,
                    best_rolls,
                    elapsed_secs,
                });
            });
            let _ = tx.send(GuiMsg::LuckyDone(outcome));
        });

        self.rx = Some(rx);
        self.stop = Some(stop);
        self.status = Status::Running;
        self.lucky_seeds = 0;
        self.lucky_rolls = 0;
        self.lucky_best = None;
        self.elapsed_secs = 0.0;
        self.log.clear();
        self.log.push(format!(
            "Seeking seed winning within {} rolls from seed {base}…",
            fmt_comma(threshold as u128),
        ));
    }

    fn fail(&mut self, msg: &str) {
        self.error = msg.to_owned();
        self.status = Status::Error;
    }

    fn stop_run(&mut self) {
        if let Some(flag) = &self.stop {
            flag.store(true, Ordering::Relaxed);
        }
    }

    fn poll_worker(&mut self) {
        // Drain pending messages first so we don't hold an immutable
        // borrow of `self.rx` while mutating the rest of `self`.
        let mut pending = Vec::new();
        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                let is_done = matches!(msg, GuiMsg::Done(_));
                pending.push(msg);
                if is_done {
                    break;
                }
            }
        }
        let mut done: Option<RollOutcome> = None;
        let mut done_lucky: Option<LuckyOutcome> = None;
        for msg in pending {
            match msg {
                GuiMsg::Event(RollEvent::Log { rolls, dice }) => {
                    self.push_log(format!("Roll {rolls:>7}: {dice:?}"));
                }
                GuiMsg::Event(RollEvent::Progress {
                    rolls,
                    elapsed_secs,
                    last,
                }) => {
                    self.rolls = rolls;
                    self.elapsed_secs = elapsed_secs;
                    self.last = last;
                }
                GuiMsg::Done(outcome) => {
                    done = Some(outcome);
                    break;
                }
                GuiMsg::LuckyProgress {
                    seeds_tried,
                    total_rolls,
                    best_rolls,
                    elapsed_secs,
                } => {
                    self.lucky_seeds = seeds_tried;
                    self.lucky_rolls = total_rolls;
                    self.lucky_best = best_rolls;
                    self.elapsed_secs = elapsed_secs;
                }
                GuiMsg::LuckyDone(outcome) => {
                    done_lucky = Some(outcome);
                    break;
                }
            }
        }
        if let Some(outcome) = done {
            self.finish(outcome);
        }
        if let Some(outcome) = done_lucky {
            self.finish_lucky(outcome);
        }
    }

    fn push_log(&mut self, line: String) {
        self.log.push(line);
        if self.log.len() > MAX_LOG_LINES {
            let overflow = self.log.len() - MAX_LOG_LINES;
            self.log.drain(..overflow);
        }
    }

    fn finish(&mut self, outcome: RollOutcome) {
        match outcome {
            RollOutcome::Won {
                rolls,
                dice,
                elapsed_secs,
            } => {
                self.status = Status::Won;
                self.rolls = rolls;
                self.elapsed_secs = elapsed_secs;
                self.last = dice.clone();
                self.push_log(format!(
                    "Winning roll #{}: {dice:?}",
                    fmt_comma(rolls as u128)
                ));
                let rate = if elapsed_secs > 0.0 {
                    format!(
                        "{} rolls/sec",
                        fmt_comma((rolls as f64 / elapsed_secs).round() as u128)
                    )
                } else {
                    "—".to_owned()
                };
                self.result = format!(
                    "Match on roll {} ({dice:?}) in {elapsed_secs:.3}s — {rate}",
                    fmt_comma(rolls as u128),
                );
            }
            RollOutcome::MaxRolls {
                rolls,
                elapsed_secs,
            } => {
                self.status = Status::MaxRolls;
                self.rolls = rolls;
                self.elapsed_secs = elapsed_secs;
                self.result = format!(
                    "Stopped after {} rolls without a match ({elapsed_secs:.1}s).",
                    fmt_comma(rolls as u128)
                );
            }
            RollOutcome::Stopped {
                rolls,
                elapsed_secs,
            } => {
                self.status = Status::Stopped;
                self.rolls = rolls;
                self.elapsed_secs = elapsed_secs;
                self.result = format!(
                    "Interrupted after {} rolls ({elapsed_secs:.1}s).",
                    fmt_comma(rolls as u128)
                );
            }
        }
        self.rx = None;
        self.stop = None;
    }

    fn finish_lucky(&mut self, outcome: LuckyOutcome) {
        match outcome {
            LuckyOutcome::Found {
                seed,
                rolls,
                dice,
                win_logs,
                seeds_tried,
                total_rolls,
                elapsed_secs,
            } => {
                self.status = Status::LuckyFound;
                self.lucky_seeds = seeds_tried;
                self.lucky_rolls = total_rolls;
                self.lucky_best = Some(rolls);
                self.elapsed_secs = elapsed_secs;
                self.replay_seed = Some(seed);
                for (g, d) in win_logs {
                    self.push_log(format!("Roll {g:>7}: {d:?}"));
                }
                self.push_log(format!(
                    "Lucky seed {seed}: wins in {} rolls",
                    fmt_comma(rolls as u128)
                ));
                self.result = format!(
                    "Lucky seed {seed}: wins in {} rolls with {dice:?} ({} seeds tried)",
                    fmt_comma(rolls as u128),
                    fmt_comma(seeds_tried as u128),
                );
            }
            LuckyOutcome::Exhausted {
                seeds_tried,
                total_rolls,
                elapsed_secs,
            } => {
                self.status = Status::Exhausted;
                self.lucky_seeds = seeds_tried;
                self.lucky_rolls = total_rolls;
                self.elapsed_secs = elapsed_secs;
                self.result = format!(
                    "No seed found after {} seeds ({} rolls).",
                    fmt_comma(seeds_tried as u128),
                    fmt_comma(total_rolls as u128),
                );
            }
            LuckyOutcome::Stopped {
                seeds_tried,
                total_rolls,
                elapsed_secs,
            } => {
                self.status = Status::Stopped;
                self.lucky_seeds = seeds_tried;
                self.lucky_rolls = total_rolls;
                self.elapsed_secs = elapsed_secs;
                self.result = format!(
                    "Interrupted after {} seeds ({} rolls, {elapsed_secs:.1}s).",
                    fmt_comma(seeds_tried as u128),
                    fmt_comma(total_rolls as u128),
                );
            }
        }
        self.rx = None;
        self.stop = None;
    }

    fn rate(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.rolls as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }
}

impl eframe::App for DiceApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_worker();

        // Live expected-rolls preview whenever not rolling: updates as
        // the user types, no need to press Roll first. This covers Idle
        // as well as post-run states (Won/MaxRolls/Stopped/Error), so
        // editing the inputs after a run refreshes the estimate while
        // the previous result stays visible.
        if !self.running() {
            if self.lucky_mode {
                match self.parse_dice() {
                    Err(msg) => {
                        self.error = msg;
                        self.expected_note.clear();
                    }
                    Ok((n, x, _)) => match self.parse_lucky() {
                        Err(msg) => {
                            self.error = msg;
                            self.expected_note.clear();
                        }
                        Ok((pct, _)) => {
                            self.error.clear();
                            let t = lucky_threshold(n, x, pct);
                            self.expected_note = format!(
                                "Need win within {} rolls ({pct}% of expected ~{}).",
                                fmt_comma(t as u128),
                                fmt_comma(expected_rolls(n, x, NO_CAP).0),
                            );
                        }
                    },
                }
            } else {
                match self.parse_inputs() {
                    Ok((n, x, _, cap)) => {
                        self.error.clear();
                        self.expected_note = Self::expected_note_for(n, x, cap);
                    }
                    Err(msg) => {
                        self.error = msg;
                        self.expected_note.clear();
                    }
                }
            }
        }

        egui::Panel::top("header").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Dice Roller");
                ui.with_layout(
                    egui::Layout::right_to_left(egui::Align::Center),
                    |ui| {
                        ui.label(format!("Status: {}", self.status.label()));
                    },
                );
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            let running = self.running();
            let lucky = self.lucky_mode;
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!running, |ui| {
                    ui.radio_value(&mut self.lucky_mode, false, "Normal");
                    ui.radio_value(&mut self.lucky_mode, true, "Lucky search")
                        .on_hover_text("Keep trying seeds until one wins within % of expected rolls.");
                });
            });
            ui.add_enabled_ui(!running, |ui| {
                egui::Grid::new("inputs")
                    .num_columns(2)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.label("Dice (n):");
                        ui.text_edit_singleline(&mut self.n_text);
                        ui.end_row();
                        ui.label("Sides per die (x):");
                        ui.text_edit_singleline(&mut self.x_text);
                        ui.end_row();
                        ui.label("Seed (optional):");
                        ui.text_edit_singleline(&mut self.seed_text)
                            .on_hover_text("Empty = random (normal) or 0 = first seed tried (lucky search). A set seed runs single-threaded for exact reproduction unless Threads is set.");
                        ui.end_row();
                        if lucky {
                            ui.label("Luck %:");
                            ui.text_edit_singleline(&mut self.pct_text)
                                .on_hover_text("Win within this % of expected rolls.");
                            ui.end_row();
                            ui.label("Max seeds (0 = unlimited):");
                            ui.text_edit_singleline(&mut self.max_seeds_text)
                                .on_hover_text("Empty or 0 = keep searching until stopped.");
                            ui.end_row();
                        } else {
                            ui.label("Max rolls (0 = unlimited):");
                            ui.text_edit_singleline(&mut self.max_text)
                                .on_hover_text("Empty or 0 = roll until a match.");
                            ui.end_row();
                        }
                        ui.label("Threads (0 = auto):");
                        ui.text_edit_singleline(&mut self.threads_text)
                            .on_hover_text("0 = auto (all CPUs, or 1 thread when a seed is set, for exact reproduction). Explicit values always win.");
                        ui.end_row();
                    });
            });

            if !self.expected_note.is_empty() {
                ui.label(&self.expected_note);
            }
            if !self.error.is_empty() {
                ui.colored_label(egui::Color32::RED, &self.error);
            }

            ui.horizontal(|ui| {
                let roll = ui.add_enabled(!running, egui::Button::new("Roll"));
                if roll.clicked() {
                    self.start_run();
                }
                let stop = ui.add_enabled(running, egui::Button::new("Stop"));
                if stop.clicked() {
                    self.stop_run();
                }
                if let Some(s) = self.replay_seed {
                    if ui
                        .add_enabled(!running, egui::Button::new("Replay seed"))
                        .on_hover_text("Load the lucky seed for a normal run.")
                        .clicked()
                    {
                        self.seed_text = s.to_string();
                        self.lucky_mode = false;
                        self.replay_seed = None;
                    }
                }
                if running {
                    ui.add(egui::Spinner::new());
                }
            });

            ui.separator();

            if lucky {
                let best = self
                    .lucky_best
                    .map(|b| fmt_comma(b as u128))
                    .unwrap_or_else(|| "—".to_owned());
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("Seeds: {}", fmt_comma(self.lucky_seeds as u128)));
                    ui.label(format!("Rolls: {}", fmt_comma(self.lucky_rolls as u128)));
                    ui.label(format!("Best: {best}"));
                    ui.label(format!("Elapsed: {:.1}s", self.elapsed_secs));
                });
            } else {
                let rate = self.rate();
                ui.horizontal_wrapped(|ui| {
                    ui.label(format!("Rolls: {}", fmt_comma(self.rolls as u128)));
                    ui.label(format!("Elapsed: {:.1}s", self.elapsed_secs));
                    if rate > 0.0 {
                        ui.label(format!(
                            "Rate: {} rolls/sec",
                            fmt_comma(rate.round() as u128)
                        ));
                    }
                });
            }
            if !self.last.is_empty() {
                ui.label(format!("Last: {:?}", self.last));
            }
            if !self.result.is_empty() {
                ui.label(egui::RichText::new(&self.result).strong());
            }

            ui.separator();
            ui.label("Sampled rolls:");
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .max_height(260.0)
                .show(ui, |ui| {
                    for line in &self.log {
                        ui.monospace(line);
                    }
                });
        });

        if self.running() {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([580.0, 660.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Dice Roller",
        options,
        Box::new(|_cc| Ok(Box::new(DiceApp::default()))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with(n: &str, x: &str, max: &str) -> DiceApp {
        DiceApp {
            n_text: n.to_owned(),
            x_text: x.to_owned(),
            seed_text: String::new(),
            max_text: max.to_owned(),
            ..DiceApp::default()
        }
    }

    #[test]
    fn huge_dice_rejected_not_hung() {
        let app = app_with("1000000000", "6", "1000");
        let err = app.parse_inputs().unwrap_err();
        assert!(err.contains("10,000"), "unexpected error: {err}");
    }

    #[test]
    fn max_boundary_accepted() {
        let app = app_with("10000", "6", "1000");
        let (n, _, _, _) = app.parse_inputs().unwrap();
        assert_eq!(n, 10_000);
    }

    #[test]
    fn empty_or_zero_max_is_unlimited() {
        for max in ["", "0", "0,000"] {
            let app = app_with("2", "6", max);
            assert_eq!(app.parse_inputs().unwrap().3, None, "max={max:?}");
        }
    }

    #[test]
    fn lucky_fields_validation() {
        let mut app = app_with("2", "6", "1000");
        app.pct_text = "10".to_owned();
        app.max_seeds_text = "".to_owned();
        assert_eq!(app.parse_lucky().unwrap(), (10.0, 0));
        app.max_seeds_text = "0".to_owned();
        assert_eq!(app.parse_lucky().unwrap(), (10.0, 0));
        app.max_seeds_text = "5,000".to_owned();
        assert_eq!(app.parse_lucky().unwrap(), (10.0, 5000));
        for bad_pct in ["", "abc", "-5", "0", "NaN", "inf"] {
            app.pct_text = bad_pct.to_owned();
            assert!(app.parse_lucky().is_err(), "pct={bad_pct:?}");
        }
        app.pct_text = "10".to_owned();
        app.max_seeds_text = "abc".to_owned();
        assert!(app.parse_lucky().is_err());
    }

    #[test]
    fn thread_resolution_prefers_exactness() {
        // Explicit values always win.
        assert_eq!(DiceApp::resolve_threads(8, Some(653)), 8);
        assert_eq!(DiceApp::resolve_threads(8, None), 8);
        assert_eq!(DiceApp::resolve_threads(1, Some(653)), 1);
        // Auto + explicit seed pins single-threaded (matches lucky eval).
        assert_eq!(DiceApp::resolve_threads(0, Some(653)), 1);
        // Auto without seed uses all CPUs.
        assert!(DiceApp::resolve_threads(0, None) >= 1);
    }
}
