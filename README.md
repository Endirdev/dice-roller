# Dice Roller

Roll `n` dice with `x` sides until all dice show the same face — plus a
lucky-seed search mode that hunts for seeds winning far faster than average.
Available as an optimized Rust CLI, an egui desktop GUI, and the original
Python script.

## Binaries

| Binary | Source | Description |
| --- | --- | --- |
| `dice` | `src/main.rs` | Terminal CLI |
| `dice-gui` | `src/bin/dice_gui.rs` | Desktop GUI (egui) |
| `dice.py` | — | Original Python implementation |

Both Rust binaries share the engine in `src/lib.rs`.

## Build & run

Requires a Rust toolchain (`rustup`). No system dependencies beyond a display
server for the GUI.

```sh
cargo build --release --bin dice        # CLI -> target/release/dice
cargo build --release --bin dice-gui    # GUI -> target/release/dice-gui
cargo test                              # unit tests (engine determinism, caps, lucky search)
```

## CLI usage

```sh
dice <N> <X> [--seed S] [--max-rolls N] [--threads T] [--lucky PCT] [--max-seeds N]
```

- `N` — number of dice (1–10,000), `X` — sides per die (≥ 2).
- `--seed S` — reproducible runs. An explicit seed pins single-threaded
  execution for exact reproduction unless `--threads` is set explicitly.
- `--max-rolls N` — stop after N rolls without a match (default
  15,000,000,000; `0` or empty = unlimited).
- `--threads T` — worker threads (`0` = auto = all CPUs). Same
  `(seed, threads)` always reproduces the same run.
- `--lucky PCT` — lucky-seed search: try seeds from `--seed` (default 0)
  until one wins within PCT% of the expected roll count. `--max-rolls` is
  ignored in this mode (per-seed cap is the threshold).
- `--max-seeds N` — cap seeds tried in `--lucky` mode (default 100,000;
  `0` = unlimited).

Example:

```sh
dice 2 6 --seed 0
dice 10 100 --threads 32
dice 5 20 --lucky 10 --threads 8
```

## GUI usage

Launch `dice-gui`. Enter dice, sides, optional seed, max rolls (empty/`0` =
unlimited), and threads (`0` = auto). The expected-rolls estimate updates
live as you type. Toggle **Lucky search** for seed hunting (Luck % + Max
seeds fields), and use **Replay seed** to load a found seed into a normal
run — the outcome reproduces exactly.

## Notes

- Expected rolls for `n` dice are `x^(n-1)`; a stderr warning appears when
  that exceeds the roll cap.
- Lucky-seed search costs roughly one expected run in total regardless of
  threshold — it finds demonstrably lucky seeds for small/medium counts,
  it doesn't beat the odds.
- `target/` build output and the compiled `dice-gui` binary are git-ignored;
  only sources are versioned.
