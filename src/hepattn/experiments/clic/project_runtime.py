"""Project a full training's wall time from the first few hundred steps of its SLURM log.

Run a short preflight (``--trainer.max_steps=300`` is enough) and point this at its log:

    python project_runtime.py slurm_logs/slurm-<jobid>.<name>.out --epochs 200

It reads the Lightning progress bar, which prints ``Epoch 0:  23%|...| 300/1295 [04:14<...``
every ``refresh_rate`` steps, fits the step rate on the samples after the first 100 steps
(so torch.compile warm-up does not drag the estimate), and prints the projected wall time
for the full run together with the SLURM ``--time`` to request (1.3x, rounded up to the
hour). It only REPORTS: the person submitting decides what to request.
"""

import argparse
import math
import re
import sys
from pathlib import Path

BAR = re.compile(r"Epoch (\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[(\d+):(\d+)(?::(\d+))?<")


def parse_samples(text: str) -> tuple[list[tuple[int, float]], int]:
    """(step, elapsed seconds) samples from epoch 0, and the number of steps per epoch."""
    samples, steps_per_epoch = [], 0
    for match in BAR.finditer(text):
        epoch, step, total, *clock = match.groups()
        if int(epoch) != 0:
            break
        parts = [int(c) for c in clock if c is not None]
        elapsed = parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else parts[0] * 60 + parts[1]
        samples.append((int(step), float(elapsed)))
        steps_per_epoch = int(total)
    return samples, steps_per_epoch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--epochs", type=int, default=200, help="max_epochs of the full run")
    parser.add_argument("--warmup-steps", type=int, default=100, help="ignore samples before this step")
    parser.add_argument("--margin", type=float, default=1.3, help="safety factor on the projection")
    parser.add_argument("--val-fraction", type=float, default=0.05, help="extra time for validation passes, as a fraction")
    args = parser.parse_args()

    samples, steps_per_epoch = parse_samples(args.log.read_text(errors="ignore"))
    usable = [s for s in samples if s[0] >= args.warmup_steps]
    if len(usable) < 2 or steps_per_epoch == 0:
        print(f"not enough progress-bar samples after step {args.warmup_steps} in {args.log} (found {len(samples)} in total)")
        return 1

    (s0, t0), (s1, t1) = usable[0], usable[-1]
    rate = (s1 - s0) / (t1 - t0)  # steps per second, steady state
    train_seconds = steps_per_epoch * args.epochs / rate
    total_seconds = train_seconds * (1 + args.val_fraction)
    request_hours = math.ceil(total_seconds * args.margin / 3600)

    print(f"log:                 {args.log}")
    print(f"steps per epoch:     {steps_per_epoch}")
    print(f"steady-state rate:   {rate:.2f} it/s  (steps {s0}-{s1}, {len(usable)} samples)")
    print(f"projected wall time: {total_seconds / 3600:.1f} h for {args.epochs} epochs (train + {args.val_fraction:.0%} validation)")
    print(f"suggested request:   --time={request_hours:02d}:00:00  ({args.margin}x, rounded up to the hour)")
    print("This is a report only. Set --time yourself when you submit the full run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
