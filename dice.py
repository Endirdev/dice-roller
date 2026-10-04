#!/usr/bin/env python3
"""Roll n dice with x sides until all dice show the same face.

Output strategy:
- A live progress line is updated in place (throttled to ~6 updates/sec)
  so the roll loop never blocks on I/O.
- Individual rolls are logged adaptively: every roll at first, with the gap
  to the next log growing in proportion to the count (total lines scale as
  ~10 * ln(count), i.e. a few dozen even for millions of rolls).
"""

import argparse
import random
import sys
import time

PROGRESS_INTERVAL = 0.15  # seconds between live progress updates


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Roll n dice with x sides until all dice match."
    )
    parser.add_argument("n", type=int, help="number of dice (>= 1)")
    parser.add_argument("x", type=int, help="sides per die (>= 2)")
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducible runs"
    )
    parser.add_argument(
        "--max-rolls",
        type=int,
        default=10_000_000,
        metavar="N",
        help="stop after N rolls if no match (default: 10,000,000)",
    )
    args = parser.parse_args(argv)

    if args.n < 1:
        parser.error("n must be >= 1")
    if args.x < 2:
        parser.error("x must be >= 2")
    return args


def main(argv=None):
    args = parse_args(argv)
    n, x = args.n, args.x

    rng = random.Random(args.seed)
    live = sys.stdout.isatty()

    expected = x ** (n - 1)
    if expected > args.max_rolls:
        print(
            f"warning: expected rolls ~{expected:,} exceeds "
            f"--max-rolls {args.max_rolls:,}",
            file=sys.stderr,
        )

    def clear_line():
        if live:
            sys.stdout.write("\r\x1b[2K")

    def progress(roll_count, elapsed, last_roll):
        sys.stdout.write(
            f"\r\x1b[2KRolls: {roll_count:,} | Elapsed: {elapsed:.1f}s"
            f" | Last: {last_roll}"
        )
        sys.stdout.flush()

    randint = rng.randint
    start = time.perf_counter()
    roll_count = 0
    next_log_at = 1  # every roll is logged until the step grows past 1
    next_update = 0.0

    try:
        while True:
            roll = [randint(1, x) for _ in range(n)]
            roll_count += 1

            if all(v == roll[0] for v in roll):
                break

            # Adaptive logging: the step to the next logged roll grows in
            # proportion to the count (count / 10), so early rolls are all
            # printed and total lines scale as ~10 * ln(count).
            if roll_count >= next_log_at:
                clear_line()
                print(f"Roll {roll_count:>7}: {roll}")
                next_log_at += max(1, roll_count // 10)

            now = time.perf_counter() if live else 0.0
            if live and now >= next_update:
                progress(roll_count, now - start, roll)
                next_update = now + PROGRESS_INTERVAL

            if roll_count >= args.max_rolls:
                clear_line()
                print(
                    f"Stopped after {args.max_rolls:,} rolls without a match.",
                    file=sys.stderr,
                )
                return 1
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - start
        clear_line()
        print(f"\nInterrupted after {roll_count:,} rolls ({elapsed:.1f}s).")
        return 130

    elapsed = time.perf_counter() - start
    clear_line()
    print(f"Winning roll #{roll_count:,}: {roll}")
    print(f"Total rolls: {roll_count:,}")
    print(f"Elapsed:     {elapsed:.3f}s")
    if elapsed > 0:
        print(f"Rate:        {roll_count / elapsed:,.0f} rolls/sec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
