"""Runs the large-sample jobs of rolling_basic.py at most 6 at a time: the basic Engression, 12 test years x 3 seeds
(longest trainings first). Logs: logs/large_basic/<job>.log; state: logs/large_basic_status.json.

Usage: python src/run_large_basic.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

NEW = Path(__file__).resolve().parents[1]
LOG = NEW / "logs" / "large_basic"
STATUS = NEW / "logs" / "large_basic_status.json"
PY = sys.executable
MAX_ALL = 6
FOLDS, CONFIRM = (2020, 2021, 2022, 2023, 2024, 2025), (2014, 2015, 2016, 2017, 2018, 2019)
ORDER = (2019, 2018, 2017, 2016, 2015, 2025, 2014, 2024, 2023, 2022, 2021, 2020)   # most training windows first
SEEDS = (17, 29, 43)


def jobs():
    return [{"name": f"basic_{y}_{s}", "argv": ["run", str(y), str(s)], "deps": [], "group": "basic"} for y in ORDER for s in SEEDS]


def main():
    J = jobs()
    names = [j["name"] for j in J]
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "done").mkdir(exist_ok=True)
    state = {n: ("done" if (LOG / "done" / n).exists() else "pending") for n in names}
    times, running = {}, {}
    t_start = time.time()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           **{k: "2" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    env.pop("NOISE_SMOKE", None)

    def save():
        STATUS.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "started": t_start, "max_parallel": MAX_ALL,
                                      "jobs": {n: {"state": state[n], "group": J[names.index(n)]["group"], **times.get(n, {})} for n in names}},
                                     ensure_ascii=False, indent=1))
    while True:
        for n, (proc, fh) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                times[n]["minutes"] = round((time.time() - times[n]["t0"]) / 60, 1)
                state[n] = "done" if proc.returncode == 0 else "failed"
                if proc.returncode == 0:
                    (LOG / "done" / n).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
                del running[n]
        for j in J:
            if state[j["name"]] == "pending" and any(state[d] in ("failed", "blocked") for d in j["deps"]):
                state[j["name"]] = "blocked"
        for j in J:
            n = j["name"]
            if state[n] == "pending" and all(state[d] == "done" for d in j["deps"]) and len(running) < MAX_ALL:
                fh = open(LOG / f"{n}.log", "w")
                running[n] = (subprocess.Popen([PY, "src/rolling_basic.py", *j["argv"]], cwd=NEW, stdout=fh, stderr=subprocess.STDOUT, env=env), fh)
                state[n] = "running"
                times[n] = {"t0": time.time(), "started": time.strftime("%H:%M:%S")}
        save()
        if not running and not any(s == "pending" for s in state.values()):
            break
        time.sleep(5)
    bad = [n for n in names if state[n] in ("failed", "blocked")]
    print({k: list(state.values()).count(k) for k in ("done", "failed", "blocked")}, "not finished:" if bad else "", bad or "")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
