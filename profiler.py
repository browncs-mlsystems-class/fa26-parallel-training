# Usage: uv run profiler.py <parallelism>  (parallelism: ddp|fsdp|model|pipeline)

import sys
import os
import stat
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ──────────────────────────────────────────────────────
WORKER_MAP = {
    "ddp":      [1, 2, 4, 8],
    "model":    [1, 2, 3],
    "fsdp":     [1, 2, 4],
    "pipeline": [1, 2, 3],
}

VALID_TYPES = set(WORKER_MAP)

LOG_DIR = os.path.join(SCRIPT_DIR, "profile_logs")
JOB_TIME_HOURS = 10


def main():
    if len(sys.argv) != 2:
        print("Usage: $0 <parallelism>")
        print(f"  parallelism: {', '.join(sorted(VALID_TYPES))}")
        sys.exit(1)

    parallelism = sys.argv[1]
    if parallelism not in VALID_TYPES:
        print(f"Unknown parallelism: {parallelism}")
        sys.exit(1)

    num_workers = WORKER_MAP[parallelism]
    max_gpus = max(num_workers)
    job_name = f"profile_{parallelism}"
    log_file = os.path.join(LOG_DIR, f"{job_name}.out")
    job_script = os.path.join(LOG_DIR, f"{job_name}.sh")

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Profile: {parallelism}")
    print(f"  Submitting: {job_name} (workers: {num_workers}, GPUs: {max_gpus})")

    # Generate SBATCH job script
    slurm_script = f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpus
#SBATCH --gres=gpu:{max_gpus}
#SBATCH --time={JOB_TIME_HOURS}:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --constraint=gtx_1080_ti|gtx_2080_ti|titan_rtx|rtx_a6000|l40

cd {SCRIPT_DIR}
for NUM_WORKER in {' '.join(str(w) for w in num_workers)}; do
  echo "=== Running with $NUM_WORKER worker(s) ==="
  uv run python3 train.py --num_workers $NUM_WORKER {parallelism}
done
"""

    with open(job_script, "w") as f:
        f.write(slurm_script)
    os.chmod(job_script, os.stat(job_script).st_mode | stat.S_IEXEC)

    result = subprocess.run(
        ["sbatch", "--parsable", job_script],
        capture_output=True, text=True, check=True,
    )
    job_id = result.stdout.strip()
    print(f"    Job ID: {job_id}")

    print()
    print(f"Monitor with: squeue -u $(whoami)")
    print(f"Logs in: {LOG_DIR}/")


if __name__ == "__main__":
    main()
