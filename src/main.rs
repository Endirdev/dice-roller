use clap::Parser;
use dice::{
    MAX_DICE, NO_CAP, expected_rolls, fmt_comma, LuckyConfig, LuckyEvent, LuckyOutcome,
    RollConfig, RollEvent, RollOutcome, find_lucky_seed, lucky_threshold, run_rolls_parallel,
};
use std::io::{BufWriter, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Roll n dice with x sides until all dice show the same face.
#[derive(Parser, Debug)]
#[command(name = "dice", version, about = "Roll n dice with x sides until all match")]
struct Args {
    /// number of dice (>= 1)
    n: usize,
    /// sides per die (>= 2)
    x: u32,
    /// random seed for reproducible runs (single-threaded unless --threads is set; first seed tried in --lucky mode)
    #[arg(long)]
    seed: Option<i64>,
    /// stop after N rolls if no match, 0 = unlimited [default: 15000000000]
    #[arg(long, value_name = "N")]
    max_rolls: Option<u64>,
    /// worker threads for rolling (0 = auto = all CPUs, except auto + explicit seed uses 1 for exact reproduction)
    #[arg(long, default_value_t = 0, value_name = "T")]
    threads: usize,
    /// search seeds until one wins within PCT percent of expected rolls
    #[arg(long, value_name = "PCT")]
    lucky: Option<f64>,
    /// max seeds to try in --lucky mode (0 = unlimited)
    #[arg(long, default_value_t = 100_000, value_name = "N")]
    max_seeds: u64,
}

const DEFAULT_MAX_ROLLS: u64 = 15_000_000_000;

fn clear_line(out: &mut impl Write, live: bool) {
    if live {
        let _ = out.write_all(b"\r\x1b[2K");
    }
}

fn real_main() -> i32 {
    let args = Args::parse();
    let n = args.n;
    let x = args.x;

    if n < 1 {
        eprintln!("error: n must be >= 1");
        return 2;
    }
    if n > MAX_DICE {
        eprintln!(
            "error: n must be <= {} (got {n})",
            fmt_comma(MAX_DICE as u128)
        );
        return 2;
    }
    if x < 2 {
        eprintln!("error: x must be >= 2");
        return 2;
    }

    let auto = || {
        std::thread::available_parallelism()
            .map(|nz| nz.get())
            .unwrap_or(1)
    };

    if let Some(pct) = args.lucky {
        // Lucky evaluation is thread-count independent: always use full
        // parallelism for the search workers.
        let threads = if args.threads == 0 { auto() } else { args.threads };
        return lucky_main(
            n,
            x,
            pct,
            args.seed,
            args.max_seeds,
            args.max_rolls.is_some(),
            threads,
        );
    }

    // An explicit seed pins single-threaded for exact reproduction
    // (matching lucky-search evaluation, so replay matches) unless
    // --threads is set explicitly.
    let threads = if args.threads != 0 {
        args.threads
    } else if args.seed.is_some() {
        1
    } else {
        auto()
    };

    // 0 = unlimited: map to the NO_CAP sentinel (practically unreachable
    // as a count) and skip the cap warning, which is meaningless then.
    let cap = args.max_rolls.unwrap_or(DEFAULT_MAX_ROLLS);
    let effective = if cap == 0 { NO_CAP } else { cap };
    let (expected, exceeds) = expected_rolls(n, x, effective);
    if cap != 0 && exceeds {
        eprintln!(
            "warning: expected rolls ~{} exceeds --max-rolls {}",
            fmt_comma(expected),
            fmt_comma(cap as u128)
        );
    }

    let live = std::io::stdout().is_terminal();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&interrupted);
        let _ = ctrlc::set_handler(move || {
            flag.store(true, Ordering::Relaxed);
        });
    }

    let config = RollConfig {
        n,
        x,
        seed: args.seed.map(|s| s as u64),
        max_rolls: effective,
        threads,
    };

    let outcome = run_rolls_parallel(&config, &interrupted, |ev| match ev {
        RollEvent::Log { rolls, dice } => {
            clear_line(&mut out, live);
            let _ = writeln!(out, "Roll {:>7}: {:?}", rolls, dice);
        }
        RollEvent::Progress {
            rolls,
            elapsed_secs,
            last,
        } => {
            if live {
                let _ = write!(
                    out,
                    "\r\x1b[2KRolls: {} | Elapsed: {:.1}s | Last: {:?}",
                    fmt_comma(rolls as u128),
                    elapsed_secs,
                    last
                );
                let _ = out.flush();
            }
        }
    });

    match outcome {
        RollOutcome::Won {
            rolls,
            dice,
            elapsed_secs,
        } => {
            clear_line(&mut out, live);
            let _ = writeln!(
                out,
                "Winning roll #{}: {:?}",
                fmt_comma(rolls as u128),
                dice
            );
            let _ = writeln!(out, "Total rolls: {}", fmt_comma(rolls as u128));
            let _ = writeln!(out, "Elapsed:     {:.3}s", elapsed_secs);
            if elapsed_secs > 0.0 {
                let rate = (rolls as f64 / elapsed_secs).round() as u64;
                let _ = writeln!(out, "Rate:        {} rolls/sec", fmt_comma(rate as u128));
            }
            let _ = out.flush();
            0
        }
        RollOutcome::MaxRolls { rolls, .. } => {
            debug_assert_eq!(rolls, effective);
            clear_line(&mut out, live);
            let _ = out.flush();
            eprintln!(
                "Stopped after {} rolls without a match.",
                fmt_comma(rolls as u128)
            );
            1
        }
        RollOutcome::Stopped { rolls, elapsed_secs } => {
            clear_line(&mut out, live);
            let _ = writeln!(
                out,
                "\nInterrupted after {} rolls ({:.1}s).",
                fmt_comma(rolls as u128),
                elapsed_secs
            );
            let _ = out.flush();
            130
        }
    }
}

/// Lucky-seed search mode: try seeds from `base` until one wins within the
/// luck threshold. The per-seed cap is the threshold itself, so an explicit
/// `--max-rolls` is ignored (with a note).
fn lucky_main(
    n: usize,
    x: u32,
    pct: f64,
    seed: Option<i64>,
    max_seeds: u64,
    max_rolls_given: bool,
    threads: usize,
) -> i32 {
    if !(pct.is_finite() && pct > 0.0) {
        eprintln!("error: --lucky must be a positive number (percent of expected rolls)");
        return 2;
    }
    if max_rolls_given {
        eprintln!(
            "note: --max-rolls is ignored in --lucky mode (per-seed cap is the luck threshold)."
        );
    }

    let (expected, _) = expected_rolls(n, x, u64::MAX);
    let threshold = lucky_threshold(n, x, pct);
    let base: u64 = seed.map(|s| s as u64).unwrap_or(0);

    let live = std::io::stdout().is_terminal();
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&interrupted);
        let _ = ctrlc::set_handler(move || {
            flag.store(true, Ordering::Relaxed);
        });
    }

    let config = LuckyConfig {
        n,
        x,
        base_seed: base,
        threshold,
        max_seeds,
        threads,
    };
    let _ = writeln!(
        out,
        "Seeking seed winning within {} rolls ({pct}% of expected ~{}) from seed {base}…",
        fmt_comma(threshold as u128),
        fmt_comma(expected),
    );
    let _ = out.flush();

    let outcome = find_lucky_seed(&config, &interrupted, |ev| {
        let LuckyEvent::Progress {
            seeds_tried,
            total_rolls,
            best_rolls,
            elapsed_secs,
        } = ev;
        if live {
            let best = best_rolls
                .map(|b| fmt_comma(b as u128))
                .unwrap_or_else(|| "—".to_owned());
            let _ = write!(
                out,
                "\r\x1b[2KSeeds: {} | Rolls: {} | Best: {} | Elapsed: {:.1}s",
                fmt_comma(seeds_tried as u128),
                fmt_comma(total_rolls as u128),
                best,
                elapsed_secs,
            );
            let _ = out.flush();
        }
    });

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
            clear_line(&mut out, live);
            for (g, d) in &win_logs {
                let _ = writeln!(out, "Roll {:>7}: {:?}", g, d);
            }
            let _ = writeln!(
                out,
                "Lucky seed {seed}: wins in {} rolls (threshold {})",
                fmt_comma(rolls as u128),
                fmt_comma(threshold as u128),
            );
            let _ = writeln!(out, "Winning dice: {dice:?}");
            let _ = writeln!(
                out,
                "Seeds tried: {} (total rolls: {})",
                fmt_comma(seeds_tried as u128),
                fmt_comma(total_rolls as u128),
            );
            let _ = writeln!(out, "Elapsed:     {elapsed_secs:.3}s");
            if elapsed_secs > 0.0 {
                let rate = (seeds_tried as f64 / elapsed_secs).round() as u64;
                let _ = writeln!(
                    out,
                    "Rate:        {} seeds/sec",
                    fmt_comma(rate as u128)
                );
            }
            let _ = out.flush();
            0
        }
        LuckyOutcome::Exhausted {
            seeds_tried,
            total_rolls,
            ..
        } => {
            clear_line(&mut out, live);
            let _ = out.flush();
            eprintln!(
                "No seed found winning within {} rolls after {} seeds ({} rolls).",
                fmt_comma(threshold as u128),
                fmt_comma(seeds_tried as u128),
                fmt_comma(total_rolls as u128),
            );
            1
        }
        LuckyOutcome::Stopped {
            seeds_tried,
            total_rolls,
            elapsed_secs,
        } => {
            clear_line(&mut out, live);
            let _ = writeln!(
                out,
                "\nInterrupted after {} seeds ({} rolls, {:.1}s).",
                fmt_comma(seeds_tried as u128),
                fmt_comma(total_rolls as u128),
                elapsed_secs,
            );
            let _ = out.flush();
            130
        }
    }
}

fn main() {
    std::process::exit(real_main());
}
