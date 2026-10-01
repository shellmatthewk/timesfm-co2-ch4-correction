"""Runs the two-station large-sample jobs of rolling_residual.py with their dependencies, at most 8 at a time:
72 residual Engression trainings (CO2 and CH4, 12 test years, 3 seeds), the corrections per test year on the TimesFM
base (no training needed) and on the residual base (after that year's 6 trainings), then the two reports.
Logs: logs/large/<job>.log; state: logs/large_status.json.

Usage: python src/run_large.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

NEW = Path(__file__).resolve().parents[1]
LOG = NEW / "logs" / "large"
STATUS = NEW / "logs" / "large_status.json"
PY = sys.executable
MAX_ALL = 8
FOLDS, CONFIRM = (2020, 2021, 2022, 2023, 2024, 2025), (2014, 2015, 2016, 2017, 2018, 2019)
SEEDS = (17, 29, 43)
JOBS = []


def job(name, argv, deps=(), group=""):
    JOBS.append({"name": name, "argv": [PY, "src/rolling_residual.py", *map(str, argv)], "deps": list(deps), "group": group})
    return name


def build():
    folds = {}
    for year in FOLDS + CONFIRM:
        tr = [job(f"train_{gas}_{year}_{seed}", ["train", gas, year, seed], group="train") for gas in ("co2", "ch4") for seed in SEEDS]
        folds[year] = [job(f"fold_tfm_{year}", ["fold", "tfm", year], group="fold"), job(f"fold_res_{year}", ["fold", "res", year], tr, group="fold")]
    job("report", ["report"], [j for y in FOLDS for j in folds[y]], group="report")
    job("report_confirm", ["report", "confirm"], [j for y in CONFIRM for j in folds[y]], group="report")


def main():
    build()
    names = [j["name"] for j in JOBS]
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG / "done").mkdir(exist_ok=True)
    state = {n: ("done" if (LOG / "done" / n).exists() else "pending") for n in names}
    times, running = {}, {}
    t_start = time.time()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}

    def save():
        STATUS.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "started": t_start, "max_parallel": MAX_ALL,
                                      "jobs": {n: {"state": state[n], "group": JOBS[names.index(n)]["group"], **times.get(n, {})} for n in names}},
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
        for j in JOBS:
            if state[j["name"]] == "pending" and any(state[d] in ("failed", "blocked") for d in j["deps"]):
                state[j["name"]] = "blocked"
        for j in JOBS:
            n = j["name"]
            if state[n] == "pending" and all(state[d] == "done" for d in j["deps"]) and len(running) < MAX_ALL:
                fh = open(LOG / f"{n}.log", "w")
                running[n] = (subprocess.Popen(j["argv"], cwd=NEW, stdout=fh, stderr=subprocess.STDOUT, env=env), fh)
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
