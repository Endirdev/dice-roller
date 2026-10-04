//! Shared dice-roll engine used by the `dice` CLI and the `dice-gui` egui app.
//!
//! Hot loop keeps the max-speed properties of the original CLI:
//! fail-fast per-die early exit, no allocation outside display iterations,
//! throttled wall-clock checks, power-of-two bitmask fast path.
//!
//! [`run_rolls_parallel`] spreads independent trials across worker threads
//! in deterministic batches: same `(seed, threads)` always reproduces the
//! exact run, while wall-clock time scales ~linearly with core count.

use rand::rngs::SmallRng;
use rand::{Rng, RngCore, SeedableRng};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Minimum gap between progress events.
pub const PROGRESS_INTERVAL: Duration = Duration::from_millis(150);

/// Sentinel for [`RollConfig::max_rolls`] meaning "no cap — roll until a
/// match or an external stop". At billions of rolls/sec, reaching it would
/// take centuries, so both runners treat it as unlimited with zero
/// special-casing in the hot loop. Constructed by mapping user input of
/// empty/`0` at the CLI/GUI boundary.
pub const NO_CAP: u64 = u64::MAX;
/// Check wall-clock / stop flag every N rolls (mask). Avoids an
/// `Instant::now()` syscall in the hot loop.
const TIME_CHECK_MASK: u64 = 511;
const STOP_CHECK_MASK: u64 = 1023;

pub fn fmt_comma(mut n: u128) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let mut groups: Vec<u16> = Vec::new();
    while n >= 1000 {
        groups.push((n % 1000) as u16);
        n /= 1000;
    }
    let mut out = n.to_string();
    for g in groups.iter().rev() {
        out.push(',');
        out.push_str(&format!("{g:03}"));
    }
    out
}

/// Compute x^(n-1) with saturation. Returns (value, exceeds_max).
pub fn expected_rolls(n: usize, x: u32, max_rolls: u64) -> (u128, bool) {
    if n <= 1 {
        return (1, 1 > max_rolls as u128);
    }
    let xu = x as u128;
    let mut exact: u128 = 1;
    let mut saturated: u128 = 1;
    let mut exceeded = false;
    for _ in 0..(n - 1) {
        saturated = saturated.saturating_mul(xu);
        if !exceeded {
            match exact.checked_mul(xu) {
                Some(v) => {
                    exact = v;
                    if exact > max_rolls as u128 {
                        exceeded = true;
                    }
                }
                None => {
                    exceeded = true;
                }
            }
        }
        // Both outputs frozen from here on (exact is abandoned once
        // exceeded; saturated can't pass u128::MAX), so stop instead of
        // looping another billion times for absurd dice counts.
        if exceeded && saturated == u128::MAX {
            break;
        }
    }
    if exceeded {
        (saturated, true)
    } else {
        (exact, false)
    }
}

#[derive(Clone, Debug)]
pub struct RollConfig {
    pub n: usize,
    pub x: u32,
    /// Already normalized to u64 (`seed_arg as u64`); `None` = OS entropy.
    pub seed: Option<u64>,
    pub max_rolls: u64,
    /// Roll cap, or [`NO_CAP`] for unlimited.
    /// Worker threads for [`run_rolls_parallel`]. `0`/`1` = single-threaded
    /// legacy path (exact [`run_rolls`] behavior). `>= 2` enables
    /// deterministic batch parallelism; the winning roll index differs
    /// from single-threaded for the same seed, but is reproducible for
    /// the same `(seed, threads)`. Frontends resolve "auto + explicit
    /// seed" to 1 so seeded runs match lucky-search evaluation.
    pub threads: usize,
}

#[derive(Clone, Debug)]
pub enum RollEvent {
    /// A sampled roll for the adaptive log (`Roll {count}: [...]`).
    Log { rolls: u64, dice: Vec<u32> },
    /// Throttled live stats update.
    Progress {
        rolls: u64,
        elapsed_secs: f64,
        last: Vec<u32>,
    },
}

#[derive(Clone, Debug)]
pub enum RollOutcome {
    Won {
        rolls: u64,
        dice: Vec<u32>,
        elapsed_secs: f64,
    },
    MaxRolls {
        rolls: u64,
        elapsed_secs: f64,
    },
    Stopped {
        rolls: u64,
        elapsed_secs: f64,
    },
}

#[inline(always)]
fn next_die(rng: &mut SmallRng, x: u32, pow2: bool, mask: u32) -> u32 {
    if pow2 {
        // Exact for powers of two: single AND + add.
        (rng.next_u32() & mask) + 1
    } else {
        rng.random_range(1..=x)
    }
}

/// Roll until all dice match, `max_rolls` is hit, or `stop` is set.
/// Emits adaptive `Log` events (`count/10` schedule) and throttled
/// `Progress` events; returns the outcome with total elapsed seconds.
///
/// Deterministic for a given `(seed, n, x, max_rolls)`: only the log
/// schedule selects full rolls, never wall-clock timing. `Progress`
/// events therefore carry the most recent sampled dice, which can lag
/// the live count on very long runs.
pub fn run_rolls(
    config: &RollConfig,
    stop: &AtomicBool,
    mut emit: impl FnMut(RollEvent),
) -> RollOutcome {
    let n = config.n;
    let x = config.x;
    let max_rolls = config.max_rolls;
    debug_assert!(n >= 1 && x >= 2);

    let mut rng = match config.seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_os_rng(),
    };

    let start = Instant::now();
    let mut next_update = start; // immediate first progress event
    let mut next_log_at: u64 = 1;
    let mut roll_count: u64 = 0;

    // Reused display buffer; only filled on log iterations and wins.
    let mut display = vec![0u32; n];
    // Latest sampled dice, echoed by progress events.
    let mut last_full: Option<Vec<u32>> = None;

    let pow2 = x.is_power_of_two();
    let mask = if pow2 { x - 1 } else { 0 };

    loop {
        roll_count += 1;

        // Low-frequency stop poll (once per 1024 rolls).
        if (roll_count & STOP_CHECK_MASK) == 0 && stop.load(Ordering::Relaxed) {
            return RollOutcome::Stopped {
                rolls: roll_count,
                elapsed_secs: start.elapsed().as_secs_f64(),
            };
        }

        if roll_count >= next_log_at {
            // Slow path: full roll so log lines are honest. Taken on a
            // deterministic schedule only, so the RNG stream (and hence
            // `--seed` reproducibility) never depends on timing.
            if stop.load(Ordering::Relaxed) {
                return RollOutcome::Stopped {
                    rolls: roll_count,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                };
            }
            if pow2 {
                for slot in display.iter_mut() {
                    *slot = (rng.next_u32() & mask) + 1;
                }
            } else {
                for slot in display.iter_mut() {
                    *slot = rng.random_range(1..=x);
                }
            }
            let first = display[0];
            let mut won = true;
            for &v in &display[1..] {
                if v != first {
                    won = false;
                    break;
                }
            }
            if won {
                return RollOutcome::Won {
                    rolls: roll_count,
                    dice: display,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                };
            }

            emit(RollEvent::Log {
                rolls: roll_count,
                dice: display.clone(),
            });
            next_log_at += std::cmp::max(1, roll_count / 10);
            last_full = Some(display.clone());

            if roll_count >= max_rolls {
                return RollOutcome::MaxRolls {
                    rolls: roll_count,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                };
            }
        } else {
            // Fast path: fail-fast scalars, no alloc, no syscall.
            let first = next_die(&mut rng, x, pow2, mask);
            let mut matched = true;
            if n > 1 {
                for _ in 1..n {
                    let v = next_die(&mut rng, x, pow2, mask);
                    if v != first {
                        matched = false;
                        break;
                    }
                }
            }
            if matched {
                display.fill(first);
                return RollOutcome::Won {
                    rolls: roll_count,
                    dice: display,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                };
            }
            if roll_count >= max_rolls {
                return RollOutcome::MaxRolls {
                    rolls: roll_count,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                };
            }
        }

        // Throttled progress event. Reads no RNG state, so timing here
        // cannot perturb the roll sequence.
        if roll_count == 1 || (roll_count & TIME_CHECK_MASK) == 0 {
            if stop.load(Ordering::Relaxed) {
                return RollOutcome::Stopped {
                    rolls: roll_count,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                };
            }
            let now = Instant::now();
            if now >= next_update {
                emit(RollEvent::Progress {
                    rolls: roll_count,
                    elapsed_secs: now.duration_since(start).as_secs_f64(),
                    last: last_full.clone().unwrap_or_default(),
                });
                next_update = now + PROGRESS_INTERVAL;
            }
        }
    }
}

/// Maximum dice per roll. Bounds the display buffer (4 bytes/die) and the
/// sampled log lines (~7 bytes/die each, up to 300 kept in the GUI); beyond
/// this, output bookkeeping alone would exhaust memory long before any
/// match is plausible.
pub const MAX_DICE: usize = 10_000;

/// Trials per deterministic batch in [`run_rolls_parallel`]. At ~300M
/// rolls/sec/thread a batch takes <1ms, so coordination overhead is
/// negligible while tail waste on early wins stays small.
const BATCH_ROLLS: u64 = 1 << 18;
/// Below this expected count, stay single-threaded: thread spawn and
/// batch plumbing cost more than the run itself.
const PARALLEL_MIN_EXPECTED: u128 = 1_000_000;
/// Sanity cap on worker threads.
const MAX_PARALLEL_THREADS: usize = 1024;

/// SplitMix64: cheap stateless hash turning `(base_seed, batch)` into
/// independent per-batch subseeds.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Precomputed adaptive log points (1-based roll counts) up to
/// `max_rolls`. Matches the `next_log_at += max(1, count/10)` schedule
/// of [`run_rolls`].
pub fn log_schedule_upto(max_rolls: u64) -> Vec<u64> {
    let mut points = Vec::new();
    let mut next: u64 = 1;
    while next <= max_rolls {
        points.push(next);
        let step = std::cmp::max(1, next / 10);
        next = match next.checked_add(step) {
            Some(v) => v,
            None => break,
        };
    }
    points
}

#[derive(Debug)]
struct BatchResult {
    batch: u64,
    trials_done: u64,
    /// Global 1-based winning roll index + dice, if this batch won.
    win: Option<(u64, Vec<u32>)>,
    /// Sampled log lines owned by this batch, ascending by global index.
    logs: Vec<(u64, Vec<u32>)>,
}

/// Process one batch of trials. `batch_start` is the 1-based global index
/// of the first trial; only trials `< batch_start + batch_len` run.
/// `schedule` is the precomputed [`log_schedule_upto`] set: trials on it
/// get full honest rolls, everything else uses the fail-fast path, so RNG
/// consumption is a pure function of `(base_seed, batch)`.
fn run_batch(
    n: usize,
    x: u32,
    pow2: bool,
    mask: u32,
    base_seed: u64,
    batch: u64,
    batch_start: u64,
    batch_len: u64,
    schedule: &[u64],
    stop: &AtomicBool,
) -> BatchResult {
    let subseed = splitmix64(
        base_seed.wrapping_add(batch.wrapping_mul(0x9E3779B97F4A7C15)),
    );
    let mut rng = SmallRng::seed_from_u64(subseed);
    let mut lp = schedule.partition_point(|&p| p < batch_start);
    let mut display = vec![0u32; n];
    let mut logs = Vec::new();
    let mut i = 0u64;
    while i < batch_len {
        if (i & STOP_CHECK_MASK) == 0 && stop.load(Ordering::Relaxed) {
            break;
        }
        let g = batch_start + i; // 1-based global roll index
        if lp < schedule.len() && schedule[lp] == g {
            if pow2 {
                for slot in display.iter_mut() {
                    *slot = (rng.next_u32() & mask) + 1;
                }
            } else {
                for slot in display.iter_mut() {
                    *slot = rng.random_range(1..=x);
                }
            }
            let first = display[0];
            if display[1..].iter().all(|&v| v == first) {
                return BatchResult {
                    batch,
                    trials_done: i + 1,
                    win: Some((g, display)),
                    logs,
                };
            }
            logs.push((g, display.clone()));
            lp += 1;
        } else {
            let first = next_die(&mut rng, x, pow2, mask);
            let mut matched = true;
            if n > 1 {
                for _ in 1..n {
                    if next_die(&mut rng, x, pow2, mask) != first {
                        matched = false;
                        break;
                    }
                }
            }
            if matched {
                display.fill(first);
                return BatchResult {
                    batch,
                    trials_done: i + 1,
                    win: Some((g, display)),
                    logs,
                };
            }
        }
        i += 1;
    }
    BatchResult {
        batch,
        trials_done: i,
        win: None,
        logs,
    }
}

/// Multi-threaded version of [`run_rolls`].
///
/// Trials are split into deterministic batches handed out via an atomic
/// counter; batch `b` draws from `splitmix64(base_seed, b)`, so the same
/// `(seed, threads)` reproduces the exact run (same winning index and log
/// sequence — only `elapsed_secs` and progress `last` vary). The
/// coordinator drains completions in batch order, so the first global win
/// terminates the run and log lines stay globally ordered.
///
/// `threads <= 1`, or runs with `expected < 1M` rolls, delegate to the
/// single-threaded [`run_rolls`] (exact legacy behavior).
pub fn run_rolls_parallel(
    config: &RollConfig,
    stop: &AtomicBool,
    mut emit: impl FnMut(RollEvent),
) -> RollOutcome {
    let n = config.n;
    let x = config.x;
    let max_rolls = config.max_rolls;
    debug_assert!(n >= 1 && x >= 2);

    let mut threads = config.threads;
    if threads <= 1 {
        return run_rolls(config, stop, emit);
    }
    if max_rolls < 1 {
        return RollOutcome::MaxRolls {
            rolls: 0,
            elapsed_secs: 0.0,
        };
    }
    let (expected, _) = expected_rolls(n, x, max_rolls);
    if expected < PARALLEL_MIN_EXPECTED {
        return run_rolls(config, stop, emit);
    }

    let num_batches = max_rolls.div_ceil(BATCH_ROLLS);
    threads = threads
        .min(num_batches as usize)
        .min(MAX_PARALLEL_THREADS)
        .max(1);
    if threads <= 1 {
        return run_rolls(config, stop, emit);
    }

    let base_seed: u64 = config.seed.unwrap_or_else(rand::random);
    let schedule = std::sync::Arc::new(log_schedule_upto(max_rolls));
    let pow2 = x.is_power_of_two();
    let mask = if pow2 { x - 1 } else { 0 };

    let start = Instant::now();
    let mut next_update = start;

    let (tx, rx) = mpsc::channel::<BatchResult>();
    let next_batch = std::sync::Arc::new(AtomicU64::new(0));
    // Lowest batch index known to contain a win; workers skip anything
    // past it since only earlier batches can still matter.
    let first_win = std::sync::Arc::new(AtomicU64::new(u64::MAX));

    let outcome = std::thread::scope(|s| {
        for _ in 0..threads {
            let tx = tx.clone();
            let next_batch = std::sync::Arc::clone(&next_batch);
            let first_win = std::sync::Arc::clone(&first_win);
            let schedule = std::sync::Arc::clone(&schedule);
            s.spawn(move || loop {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                let b = next_batch.fetch_add(1, Ordering::SeqCst);
                if b >= num_batches {
                    return;
                }
                if b > first_win.load(Ordering::SeqCst) {
                    return;
                }
                let batch_start = b * BATCH_ROLLS + 1;
                let batch_len = (max_rolls - (batch_start - 1)).min(BATCH_ROLLS);
                let r = run_batch(
                    n,
                    x,
                    pow2,
                    mask,
                    base_seed,
                    b,
                    batch_start,
                    batch_len,
                    &schedule,
                    stop,
                );
                if r.win.is_some() {
                    first_win.fetch_min(b, Ordering::SeqCst);
                    let _ = tx.send(r);
                    return;
                }
                if tx.send(r).is_err() {
                    return; // coordinator gone
                }
            });
        }
        drop(tx);

        let mut buffered: BTreeMap<u64, BatchResult> = BTreeMap::new();
        let mut next_expected: u64 = 0;
        let mut global_done: u64 = 0;
        let mut last_dice: Vec<u32> = Vec::new();

        loop {
            // Drain in-order completions; only an in-order win counts,
            // guaranteeing the earliest global win terminates the run.
            while let Some(r) = buffered.remove(&next_expected) {
                for (g, dice) in r.logs {
                    last_dice = dice.clone();
                    emit(RollEvent::Log { rolls: g, dice });
                }
                global_done += r.trials_done;
                if let Some((w, dice)) = r.win {
                    stop.store(true, Ordering::SeqCst);
                    return RollOutcome::Won {
                        rolls: w,
                        dice,
                        elapsed_secs: start.elapsed().as_secs_f64(),
                    };
                }
                next_expected += 1;
                if next_expected == num_batches {
                    return RollOutcome::MaxRolls {
                        rolls: max_rolls,
                        elapsed_secs: start.elapsed().as_secs_f64(),
                    };
                }
            }

            if stop.load(Ordering::Relaxed) {
                // External stop: every started batch reports exactly one
                // (possibly partial) result, so draining until all senders
                // are gone yields the exact attempted-trial count.
                let mut total =
                    global_done + buffered.values().map(|r| r.trials_done).sum::<u64>();
                for r in &rx {
                    total += r.trials_done;
                }
                return RollOutcome::Stopped {
                    rolls: total,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                };
            }

            let now = Instant::now();
            if now >= next_update {
                emit(RollEvent::Progress {
                    rolls: global_done,
                    elapsed_secs: now.duration_since(start).as_secs_f64(),
                    last: last_dice.clone(),
                });
                next_update = now + PROGRESS_INTERVAL;
            }
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(r) => {
                    buffered.insert(r.batch, r);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Workers only vanish early on stop (handled above) or
                    // after a win (returned above); anything else would hang,
                    // so account for what we have and stop instead.
                    let total =
                        global_done + buffered.values().map(|r| r.trials_done).sum::<u64>();
                    return RollOutcome::Stopped {
                        rolls: total,
                        elapsed_secs: start.elapsed().as_secs_f64(),
                    };
                }
            }
        }
    });
    outcome
}

// ==================== Lucky-seed search ====================

/// Seeds evaluated per task chunk in [`find_lucky_seed`]. Amortizes mpsc
/// overhead for tiny thresholds (a threshold-1 candidate is ~1 roll).
const SEED_CHUNK: u64 = 64;

/// Luck threshold: a seed qualifies if it wins within `pct` percent of the
/// expected roll count (minimum 1 roll). Requires finite `pct > 0`; larger
/// values saturate harmlessly toward "any win".
pub fn lucky_threshold(n: usize, x: u32, pct: f64) -> u64 {
    debug_assert!(pct.is_finite() && pct > 0.0);
    let (expected, _) = expected_rolls(n, x, u64::MAX);
    let t = (expected as f64 * pct / 100.0).ceil() as u128;
    t.clamp(1, u64::MAX as u128) as u64
}

#[derive(Clone, Copy, Debug)]
pub struct LuckyConfig {
    pub n: usize,
    pub x: u32,
    /// First seed tried; candidates are base, base+1, … (wrapping).
    pub base_seed: u64,
    /// A seed qualifies if it wins within this many rolls (>= 1).
    pub threshold: u64,
    /// Max seeds to try; 0 = unlimited.
    pub max_seeds: u64,
    /// Search workers. The result is identical for any thread count: task
    /// contents are fixed per chunk index and completions drain in order.
    pub threads: usize,
}

#[derive(Clone, Debug)]
pub enum LuckyEvent {
    /// Throttled search progress. `best_rolls` is the fewest rolls any
    /// evaluated seed needed so far (`None` = none finished yet).
    Progress {
        seeds_tried: u64,
        total_rolls: u64,
        best_rolls: Option<u64>,
        elapsed_secs: f64,
    },
}

#[derive(Clone, Debug)]
pub enum LuckyOutcome {
    Found {
        seed: u64,
        rolls: u64,
        dice: Vec<u32>,
        /// Sampled log lines of the winning run, ascending by roll index.
        win_logs: Vec<(u64, Vec<u32>)>,
        seeds_tried: u64,
        total_rolls: u64,
        elapsed_secs: f64,
    },
    Exhausted {
        seeds_tried: u64,
        total_rolls: u64,
        elapsed_secs: f64,
    },
    Stopped {
        seeds_tried: u64,
        total_rolls: u64,
        elapsed_secs: f64,
    },
}

#[derive(Debug)]
struct SeedChunkResult {
    chunk: u64,
    seeds_done: u64,
    rolls_done: u64,
    best: Option<u64>,
    /// Winning seed, its roll count, dice, and sampled log lines.
    win: Option<(u64, u64, Vec<u32>, Vec<(u64, Vec<u32>)>)>,
}

/// Evaluate one chunk of candidate seeds in order, stopping at the first
/// seed that wins within `threshold` (per-seed cap). Deterministic in
/// `(base_seed, chunk)`; the single-threaded [`run_rolls`] per candidate
/// keeps each seed's roll count canonical.
fn run_seed_chunk(
    n: usize,
    x: u32,
    threshold: u64,
    base_seed: u64,
    chunk: u64,
    len: u64,
    stop: &AtomicBool,
) -> SeedChunkResult {
    let mut seeds_done = 0u64;
    let mut rolls_done = 0u64;
    let mut best: Option<u64> = None;
    for j in 0..len {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let seed = base_seed.wrapping_add(chunk.wrapping_mul(SEED_CHUNK).wrapping_add(j));
        let cfg = RollConfig {
            n,
            x,
            seed: Some(seed),
            max_rolls: threshold,
            threads: 0,
        };
        let mut logs = Vec::new();
        match run_rolls(&cfg, stop, |ev| {
            if let RollEvent::Log { rolls, dice } = ev {
                logs.push((rolls, dice));
            }
        }) {
            RollOutcome::Won { rolls, dice, .. } => {
                seeds_done += 1;
                rolls_done += rolls;
                best = Some(best.map_or(rolls, |b: u64| b.min(rolls)));
                return SeedChunkResult {
                    chunk,
                    seeds_done,
                    rolls_done,
                    best,
                    win: Some((seed, rolls, dice, logs)),
                };
            }
            RollOutcome::MaxRolls { rolls, .. } => {
                seeds_done += 1;
                rolls_done += rolls;
                best = Some(best.map_or(rolls, |b: u64| b.min(rolls)));
            }
            RollOutcome::Stopped { rolls, .. } => {
                // Aborted mid-seed: count the attempted rolls, but not a
                // completed evaluation. The coordinator sees the stop flag.
                rolls_done += rolls;
                break;
            }
        }
    }
    SeedChunkResult {
        chunk,
        seeds_done,
        rolls_done,
        best,
        win: None,
    }
}

/// Search seeds until one wins within `threshold` rolls.
///
/// Candidates `base_seed, base_seed+1, …` are evaluated in fixed 64-seed
/// chunks handed out via an atomic counter; completions drain in chunk
/// order, so the earliest winning seed terminates the run and the result
/// (winning seed, roll count, log lines) is identical for any thread
/// count — only `elapsed_secs` varies. Each candidate runs the canonical
/// single-threaded [`run_rolls`] capped at `threshold`, so seeds that
/// cannot qualify abort early.
pub fn find_lucky_seed(
    config: &LuckyConfig,
    stop: &AtomicBool,
    mut emit: impl FnMut(LuckyEvent),
) -> LuckyOutcome {
    let LuckyConfig {
        n,
        x,
        base_seed,
        threshold,
        max_seeds,
        threads,
    } = *config;
    debug_assert!(n >= 1 && x >= 2 && threshold >= 1);

    let mut threads = threads.max(1).min(MAX_PARALLEL_THREADS);
    let bounded = max_seeds != 0;
    let num_chunks: u64 = if bounded {
        max_seeds.div_ceil(SEED_CHUNK)
    } else {
        u64::MAX
    };
    threads = threads.min(num_chunks as usize).max(1);

    let start = Instant::now();
    let mut next_update = start;

    let (tx, rx) = mpsc::channel::<SeedChunkResult>();
    let next_chunk = std::sync::Arc::new(AtomicU64::new(0));
    // Lowest chunk index known to contain a win; chunks past it can never
    // matter since only earlier chunks are still outstanding.
    let first_win = std::sync::Arc::new(AtomicU64::new(u64::MAX));

    let outcome = std::thread::scope(|s| {
        for _ in 0..threads {
            let tx = tx.clone();
            let next_chunk = std::sync::Arc::clone(&next_chunk);
            let first_win = std::sync::Arc::clone(&first_win);
            s.spawn(move || loop {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                let c = next_chunk.fetch_add(1, Ordering::SeqCst);
                if c >= num_chunks {
                    return;
                }
                if c > first_win.load(Ordering::SeqCst) {
                    return;
                }
                let len = if bounded {
                    // c < num_chunks guarantees c*64 < max_seeds.
                    (max_seeds - c.wrapping_mul(SEED_CHUNK)).min(SEED_CHUNK)
                } else {
                    SEED_CHUNK
                };
                let r = run_seed_chunk(n, x, threshold, base_seed, c, len, stop);
                if r.win.is_some() {
                    first_win.fetch_min(c, Ordering::SeqCst);
                    let _ = tx.send(r);
                    return;
                }
                if tx.send(r).is_err() {
                    return; // coordinator gone
                }
            });
        }
        drop(tx);

        let mut buffered: BTreeMap<u64, SeedChunkResult> = BTreeMap::new();
        let mut next_expected: u64 = 0;
        let mut seeds_done: u64 = 0;
        let mut rolls_done: u64 = 0;
        let mut best: Option<u64> = None;

        loop {
            // Drain in-order; only an in-order win counts, guaranteeing
            // the earliest winning seed terminates the run.
            while let Some(r) = buffered.remove(&next_expected) {
                seeds_done += r.seeds_done;
                rolls_done += r.rolls_done;
                if let Some(b) = r.best {
                    best = Some(best.map_or(b, |o: u64| o.min(b)));
                }
                if let Some((seed, rolls, dice, logs)) = r.win {
                    stop.store(true, Ordering::SeqCst);
                    return LuckyOutcome::Found {
                        seed,
                        rolls,
                        dice,
                        win_logs: logs,
                        seeds_tried: seeds_done,
                        total_rolls: rolls_done,
                        elapsed_secs: start.elapsed().as_secs_f64(),
                    };
                }
                next_expected += 1;
                if bounded && next_expected == num_chunks {
                    return LuckyOutcome::Exhausted {
                        seeds_tried: seeds_done,
                        total_rolls: rolls_done,
                        elapsed_secs: start.elapsed().as_secs_f64(),
                    };
                }
            }

            if stop.load(Ordering::Relaxed) {
                // External stop: every started chunk reports exactly one
                // (possibly partial) result, so draining until all senders
                // are gone yields exact attempted counts.
                let mut s = seeds_done;
                let mut t = rolls_done;
                for r in buffered.values() {
                    s += r.seeds_done;
                    t += r.rolls_done;
                }
                for r in &rx {
                    s += r.seeds_done;
                    t += r.rolls_done;
                }
                return LuckyOutcome::Stopped {
                    seeds_tried: s,
                    total_rolls: t,
                    elapsed_secs: start.elapsed().as_secs_f64(),
                };
            }

            let now = Instant::now();
            if now >= next_update {
                emit(LuckyEvent::Progress {
                    seeds_tried: seeds_done,
                    total_rolls: rolls_done,
                    best_rolls: best,
                    elapsed_secs: now.duration_since(start).as_secs_f64(),
                });
                next_update = now + PROGRESS_INTERVAL;
            }
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(r) => {
                    if let Some(b) = r.best {
                        best = Some(best.map_or(b, |o: u64| o.min(b)));
                    }
                    buffered.insert(r.chunk, r);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Workers only vanish early on stop (handled above) or
                    // after a win (returned above); account and stop rather
                    // than hang on anything unexpected.
                    let mut s = seeds_done;
                    let mut t = rolls_done;
                    for r in buffered.values() {
                        s += r.seeds_done;
                        t += r.rolls_done;
                    }
                    return LuckyOutcome::Stopped {
                        seeds_tried: s,
                        total_rolls: t,
                        elapsed_secs: start.elapsed().as_secs_f64(),
                    };
                }
            }
        }
    });
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_small() {
        assert_eq!(fmt_comma(0), "0");
        assert_eq!(fmt_comma(999), "999");
        assert_eq!(fmt_comma(1000), "1,000");
        assert_eq!(fmt_comma(1_234_567), "1,234,567");
        assert_eq!(fmt_comma(10_000_000), "10,000,000");
    }

    #[test]
    fn expected_basic() {
        assert_eq!(expected_rolls(1, 6, 10_000_000).0, 1);
        assert_eq!(expected_rolls(2, 6, 10_000_000).0, 6);
        assert_eq!(expected_rolls(3, 6, 10_000_000).0, 36);
        let (_, exceeds) = expected_rolls(20, 6, 10_000_000);
        assert!(exceeds);
    }

    #[test]
    fn expected_overflow_saturates() {
        let (v, exceeds) = expected_rolls(200, 1000, 10_000_000);
        assert!(exceeds);
        assert_eq!(v, u128::MAX);
    }

    #[test]
    fn engine_n1_wins_first_roll() {
        let stop = AtomicBool::new(false);
        let cfg = RollConfig {
            n: 1,
            x: 6,
            seed: Some(0),
            max_rolls: 100,
            threads: 0,
        };
        match run_rolls(&cfg, &stop, |_| {}) {
            RollOutcome::Won { rolls, dice, .. } => {
                assert_eq!(rolls, 1);
                assert_eq!(dice.len(), 1);
            }
            other => panic!("expected win, got {other:?}"),
        }
    }

    #[test]
    fn engine_seeded_run_is_deterministic() {
        let run = || {
            let stop = AtomicBool::new(false);
            let cfg = RollConfig {
                n: 2,
                x: 6,
                seed: Some(12345),
                max_rolls: 10_000_000,
                threads: 0,
            };
            match run_rolls(&cfg, &stop, |_| {}) {
                RollOutcome::Won { rolls, dice, .. } => (rolls, dice),
                other => panic!("expected win, got {other:?}"),
            }
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn engine_max_rolls_caps_hard_case() {
        let stop = AtomicBool::new(false);
        let cfg = RollConfig {
            n: 10,
            x: 100,
            seed: Some(7),
            max_rolls: 1000,
            threads: 0,
        };
        match run_rolls(&cfg, &stop, |_| {}) {
            RollOutcome::MaxRolls { rolls, .. } => assert_eq!(rolls, 1000),
            other => panic!("expected cap, got {other:?}"),
        }
    }

    #[test]
    fn engine_long_run_is_deterministic() {
        // Long enough to cross progress-tick boundaries: the roll
        // sequence and log events must still be identical across runs.
        let run = || {
            let stop = AtomicBool::new(false);
            let cfg = RollConfig {
                n: 10,
                x: 100,
                seed: Some(99),
                max_rolls: 8_000_000,
                threads: 0,
            };
            let mut logs = Vec::new();
            let outcome = run_rolls(&cfg, &stop, |ev| {
                if let RollEvent::Log { rolls, dice } = ev {
                    logs.push((rolls, dice));
                }
            });
            (outcome, logs)
        };
        let (o1, l1) = run();
        let (o2, l2) = run();
        // Compare rolls only: `elapsed_secs` legitimately varies.
        let rolls_of = |o: &RollOutcome| match o {
            RollOutcome::Won { rolls, .. }
            | RollOutcome::MaxRolls { rolls, .. }
            | RollOutcome::Stopped { rolls, .. } => *rolls,
        };
        assert_eq!(rolls_of(&o1), 8_000_000);
        assert_eq!(rolls_of(&o1), rolls_of(&o2));
        assert!(!l1.is_empty());
        assert_eq!(l1, l2);
    }

    #[test]
    fn engine_stop_flag_aborts() {
        let stop = AtomicBool::new(true);
        let cfg = RollConfig {
            n: 10,
            x: 100,
            seed: Some(7),
            max_rolls: 15_000_000_000,
            threads: 0,
        };
        match run_rolls(&cfg, &stop, |_| {}) {
            RollOutcome::Stopped { .. } => {}
            other => panic!("expected stop, got {other:?}"),
        }
    }

    fn outcome_rolls(o: &RollOutcome) -> u64 {
        match o {
            RollOutcome::Won { rolls, .. }
            | RollOutcome::MaxRolls { rolls, .. }
            | RollOutcome::Stopped { rolls, .. } => *rolls,
        }
    }

    #[test]
    fn parallel_one_thread_matches_single() {
        let single = RollConfig {
            n: 3,
            x: 8,
            seed: Some(5),
            max_rolls: 10_000_000,
            threads: 0,
        };
        let par = RollConfig { threads: 1, ..single.clone() };
        let stop = AtomicBool::new(false);
        let (r1, d1) = match run_rolls(&single, &stop, |_| {}) {
            RollOutcome::Won { rolls, dice, .. } => (rolls, dice),
            other => panic!("expected win, got {other:?}"),
        };
        let stop = AtomicBool::new(false);
        let (r2, d2) = match run_rolls_parallel(&par, &stop, |_| {}) {
            RollOutcome::Won { rolls, dice, .. } => (rolls, dice),
            other => panic!("expected win, got {other:?}"),
        };
        assert_eq!((r1, d1), (r2, d2));
    }

    #[test]
    fn parallel_small_expected_stays_single() {
        // 2d6 (expected 6) with threads=32 must take the legacy path.
        let cfg = RollConfig {
            n: 2,
            x: 6,
            seed: Some(0),
            max_rolls: 10_000_000,
            threads: 32,
        };
        let stop = AtomicBool::new(false);
        let from_par = match run_rolls_parallel(&cfg, &stop, |_| {}) {
            RollOutcome::Won { rolls, dice, .. } => (rolls, dice),
            other => panic!("expected win, got {other:?}"),
        };
        let single = RollConfig { threads: 0, ..cfg };
        let stop = AtomicBool::new(false);
        let from_single = match run_rolls(&single, &stop, |_| {}) {
            RollOutcome::Won { rolls, dice, .. } => (rolls, dice),
            other => panic!("expected win, got {other:?}"),
        };
        assert_eq!(from_par, from_single);
    }

    #[test]
    fn parallel_multithread_is_deterministic() {
        let run = || {
            let stop = AtomicBool::new(false);
            let cfg = RollConfig {
                n: 10,
                x: 100,
                seed: Some(99),
                max_rolls: 2_000_000,
                threads: 4,
            };
            let mut logs = Vec::new();
            let outcome = run_rolls_parallel(&cfg, &stop, |ev| {
                if let RollEvent::Log { rolls, dice } = ev {
                    logs.push((rolls, dice));
                }
            });
            (outcome, logs)
        };
        let (o1, l1) = run();
        let (o2, l2) = run();
        assert_eq!(outcome_rolls(&o1), 2_000_000);
        assert_eq!(outcome_rolls(&o1), outcome_rolls(&o2));
        assert!(!l1.is_empty());
        assert_eq!(l1, l2);
    }

    #[test]
    fn parallel_log_schedule_matches_single() {
        // On a capped run the emitted log indices must equal the
        // precomputed adaptive schedule exactly.
        let stop = AtomicBool::new(false);
        let cfg = RollConfig {
            n: 10,
            x: 100,
            seed: Some(99),
            max_rolls: 2_000_000,
            threads: 8,
        };
        let mut indices = Vec::new();
        let outcome = run_rolls_parallel(&cfg, &stop, |ev| {
            if let RollEvent::Log { rolls, .. } = ev {
                indices.push(rolls);
            }
        });
        assert!(matches!(
            outcome,
            RollOutcome::MaxRolls { rolls: 2_000_000, .. }
        ));
        assert_eq!(indices, log_schedule_upto(2_000_000));
    }

    #[test]
    fn parallel_win_path_reports_earliest_match() {
        // 4d100 (expected exactly 1M) takes the parallel path and must
        // report a real match, deterministically.
        let run = || {
            let stop = AtomicBool::new(false);
            let cfg = RollConfig {
                n: 4,
                x: 100,
                seed: Some(11),
                max_rolls: 20_000_000,
                threads: 4,
            };
            run_rolls_parallel(&cfg, &stop, |_| {})
        };
        let o1 = run();
        let o2 = run();
        for o in [&o1, &o2] {
            match o {
                RollOutcome::Won { rolls, dice, .. } => {
                    assert!(*rolls > 0);
                    assert!(dice.windows(2).all(|w| w[0] == w[1]));
                }
                other => panic!("expected win, got {other:?}"),
            }
        }
        assert_eq!(outcome_rolls(&o1), outcome_rolls(&o2));
    }

    #[test]
    fn parallel_stop_flag_aborts() {
        let stop = AtomicBool::new(true);
        let cfg = RollConfig {
            n: 10,
            x: 100,
            seed: Some(7),
            max_rolls: 15_000_000_000,
            threads: 8,
        };
        match run_rolls_parallel(&cfg, &stop, |_| {}) {
            RollOutcome::Stopped { .. } => {}
            other => panic!("expected stop, got {other:?}"),
        }
    }

    #[test]
    fn no_cap_schedule_terminates() {
        // The unbounded sentinel must still yield a finite schedule
        // (geometric ~10% growth reaches u64::MAX in a few hundred steps).
        let pts = log_schedule_upto(NO_CAP);
        assert_eq!(pts[0], 1);
        assert!(pts.len() < 1000);
    }

    #[test]
    fn engine_no_cap_easy_win() {
        let stop = AtomicBool::new(false);
        let cfg = RollConfig {
            n: 2,
            x: 6,
            seed: Some(0),
            max_rolls: NO_CAP,
            threads: 0,
        };
        match run_rolls(&cfg, &stop, |_| {}) {
            RollOutcome::Won { rolls, dice, .. } => {
                assert!(rolls > 0);
                assert!(dice.windows(2).all(|w| w[0] == w[1]));
            }
            other => panic!("expected win, got {other:?}"),
        }
    }

    #[test]
    fn expected_huge_n_returns_instantly() {
        // Without the early-break this loops a billion times and hangs
        // the GUI preview, which recomputes every frame.
        let (v, exceeds) = expected_rolls(1_000_000_000, 6, 10_000_000);
        assert!(exceeds);
        assert_eq!(v, u128::MAX);
    }

    #[test]
    fn parallel_no_cap_win_terminates() {
        // An unlimited parallel run must still terminate on a match,
        // at the same roll as the capped run with the same seed.
        let run = |max_rolls| {
            let stop = AtomicBool::new(false);
            let cfg = RollConfig {
                n: 4,
                x: 100,
                seed: Some(11),
                max_rolls,
                threads: 4,
            };
            run_rolls_parallel(&cfg, &stop, |_| {})
        };
        let uncapped = run(NO_CAP);
        let capped = run(20_000_000);
        for o in [&uncapped, &capped] {
            match o {
                RollOutcome::Won { dice, .. } => {
                    assert!(dice.windows(2).all(|w| w[0] == w[1]));
                }
                other => panic!("expected win, got {other:?}"),
            }
        }
        assert_eq!(outcome_rolls(&uncapped), outcome_rolls(&capped));
    }

    #[test]
    fn lucky_threshold_math() {
        // 2d6: expected 6.
        assert_eq!(lucky_threshold(2, 6, 100.0), 6);
        assert_eq!(lucky_threshold(2, 6, 10.0), 1); // ceil(0.6), min 1
        assert_eq!(lucky_threshold(1, 6, 5.0), 1);
        // 5d20: expected 160,000; 10% -> 16,000.
        assert_eq!(lucky_threshold(5, 20, 10.0), 16_000);
    }

    #[test]
    fn lucky_search_deterministic_across_threads() {
        // Same (base, threshold) must find the same seed on 1 and 8 workers.
        let run = |threads| {
            let stop = AtomicBool::new(false);
            let cfg = LuckyConfig {
                n: 2,
                x: 6,
                base_seed: 0,
                threshold: lucky_threshold(2, 6, 100.0),
                max_seeds: 1000,
                threads,
            };
            find_lucky_seed(&cfg, &stop, |_| {})
        };
        let one = run(1);
        let eight = run(8);
        for o in [&one, &eight] {
            match o {
                LuckyOutcome::Found {
                    rolls, dice, seeds_tried, win_logs, ..
                } => {
                    assert!(*rolls <= 6);
                    assert!(dice.windows(2).all(|w| w[0] == w[1]));
                    assert!(*seeds_tried > 0);
                    assert!(win_logs.windows(2).all(|w| w[0].0 < w[1].0));
                }
                other => panic!("expected found, got {other:?}"),
            }
        }
        let key = |o: &LuckyOutcome| match o {
            LuckyOutcome::Found { seed, rolls, .. } => (*seed, *rolls),
            _ => unreachable!(),
        };
        assert_eq!(key(&one), key(&eight));
    }

    #[test]
    fn lucky_search_exhaustion() {
        // 2d6 seed 0 wins in 4, but threshold 1 with a single seed to try.
        let stop = AtomicBool::new(false);
        let cfg = LuckyConfig {
            n: 2,
            x: 6,
            base_seed: 0,
            threshold: 1,
            max_seeds: 1,
            threads: 4,
        };
        match find_lucky_seed(&cfg, &stop, |_| {}) {
            LuckyOutcome::Exhausted {
                seeds_tried,
                total_rolls,
                ..
            } => {
                assert_eq!(seeds_tried, 1);
                assert_eq!(total_rolls, 1);
            }
            other => panic!("expected exhausted, got {other:?}"),
        }
    }

    #[test]
    fn lucky_search_unlimited_finds_win() {
        let run = || {
            let stop = AtomicBool::new(false);
            let cfg = LuckyConfig {
                n: 3,
                x: 6,
                base_seed: 0,
                threshold: lucky_threshold(3, 6, 50.0),
                max_seeds: 0,
                threads: 4,
            };
            find_lucky_seed(&cfg, &stop, |_| {})
        };
        let o1 = run();
        let o2 = run();
        for o in [&o1, &o2] {
            match o {
                LuckyOutcome::Found { rolls, dice, .. } => {
                    assert!(*rolls <= 18); // 50% of expected 36
                    assert!(dice.windows(2).all(|w| w[0] == w[1]));
                }
                other => panic!("expected found, got {other:?}"),
            }
        }
        let key = |o: &LuckyOutcome| match o {
            LuckyOutcome::Found { seed, rolls, .. } => (*seed, *rolls),
            _ => unreachable!(),
        };
        assert_eq!(key(&o1), key(&o2));
    }

    #[test]
    fn lucky_search_stop_flag_aborts() {
        let stop = AtomicBool::new(true);
        let cfg = LuckyConfig {
            n: 10,
            x: 100,
            base_seed: 0,
            threshold: 1000,
            max_seeds: 0,
            threads: 8,
        };
        match find_lucky_seed(&cfg, &stop, |_| {}) {
            LuckyOutcome::Stopped { .. } => {}
            other => panic!("expected stopped, got {other:?}"),
        }
    }
}
