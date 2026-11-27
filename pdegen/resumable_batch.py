"""Process-isolated solver batch with a durable per-attempt coordinator journal."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import cache
import json
import os
from pathlib import Path
import sqlite3
from time import perf_counter

from .adapter import check_release
from .anisotropic_bloch import solve_roll
from .audit import audit_batch
from .batch_workflow import (VALIDATORS, candidates, load_reviews, read_json,
                             review_status, solve_e3)
from .blind_prompt import render_problem
from .compare_a1_d2 import screen
from .contract import digest
from .delay_feedback import solve_shifted_delay
from .hamiltonian_polarization import solve_polarization
from .label_pilot import SEED, a1_result, readiness
from .parametric_floquet import solve_parametric
from .switched_floquet import solve_switch
from .workflow import bundled_solver


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def encode(obj):
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


@contextmanager
def controller_lock(out):
    """An OS-held lock is released when the coordinator exits or is killed."""
    with (out / ".controller.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("another coordinator is running for this output folder") from error
        yield


@cache
def delay_solver():
    return bundled_solver()[2].analyse


def attempt_path(out, ordinal, number):
    return out / "internal" / "attempts" / f"{ordinal:04d}" / f"{number:03d}"


def worker(item, out, ordinal, number, legacy):
    """Write only within this worker's immutable attempt directory."""
    folder = attempt_path(out, ordinal, number)
    start = perf_counter()
    family = item["family_id"]
    solvers = {"A1": lambda: a1_result(item, legacy, folder / "native"),
               "B3": lambda: solve_roll(item), "C2": lambda: solve_polarization(item),
               "D2": lambda: solve_shifted_delay(item, delay_solver()),
               "E3": lambda: solve_e3(item), "F3": lambda: solve_switch(item),
               "F4": lambda: solve_parametric(item)}

    def solve(_):
        try:
            return solvers[family]()
        except Exception as error:
            return dict(label="UNCERTAIN", open_rhp_count=None, evidence_grade=None,
                        solver_status="FAILED", reason=f"{type(error).__name__}: {error}")

    entry, = audit_batch([item], folder, VALIDATORS[family], solve)
    audit = read_json(folder / entry["audit"])
    result = read_json(folder / entry["result"]) if entry["result"] else {}
    if entry["result"] and family in ("A1", "D2"):
        gate, difficulty, _ = screen(item, result, audit)
    elif entry["result"]:
        gate, difficulty = readiness(item, result, audit)
    else:
        gate, difficulty = audit["status"], "NOT_SCREENED"
    relative = folder.relative_to(out)
    receipt = dict(ordinal=ordinal, attempt=number, specification_hash=item["specification_hash"],
                   worker_pid=os.getpid(), elapsed_seconds=perf_counter()-start, finished_at=now(),
                   failed=result.get("solver_status") == "FAILED", reason=result.get("reason", audit["reason"]),
                   label=result.get("label"), evidence_grade=result.get("evidence_grade"),
                   label_gate=gate, preliminary_difficulty=difficulty, sampling_group=audit["sampling_group"],
                   audit=(relative / entry["audit"]).as_posix(),
                   instance=(relative / entry["instance"]).as_posix() if entry["instance"] else None,
                   evidence=(relative / entry["result"]).as_posix() if entry["result"] else None)
    atomic_text(folder / "receipt.json", encode(receipt))


def initialize(db, items):
    db.executescript("""
    CREATE TABLE IF NOT EXISTS config (signature TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS jobs (
        ordinal INTEGER PRIMARY KEY, item TEXT NOT NULL, duplicate_of INTEGER,
        state TEXT NOT NULL, receipt TEXT);
    CREATE TABLE IF NOT EXISTS attempts (
        ordinal INTEGER NOT NULL, number INTEGER NOT NULL, state TEXT NOT NULL,
        started_at TEXT NOT NULL, finished_at TEXT, reason TEXT, receipt TEXT,
        PRIMARY KEY (ordinal, number));
    """)
    signature = digest(dict(version="resume-0.1", items=items))
    existing = db.execute("SELECT signature FROM config").fetchone()
    if existing:
        if existing[0] != signature:
            raise ValueError("resume plan or seed differs from the original batch")
        return
    seen = {}
    with db:
        db.execute("INSERT INTO config VALUES (?)", (signature,))
        for ordinal, item in enumerate(items, 1):
            key = item["specification_hash"]
            repeat = seen.get(key)
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, NULL)",
                       (ordinal, encode(item), repeat, "DUPLICATE" if repeat else "PENDING"))
            seen.setdefault(key, ordinal)


def failure_count(db, ordinal):
    return db.execute("SELECT count(*) FROM attempts WHERE ordinal=? AND state='FAILED'", (ordinal,)).fetchone()[0]


def finish(db, ordinal, number, state, reason, retries, receipt=None):
    encoded = encode(receipt) if receipt is not None else None
    with db:
        db.execute("UPDATE attempts SET state=?, finished_at=?, reason=?, receipt=? WHERE ordinal=? AND number=?",
                   (state, receipt["finished_at"] if receipt else now(), reason, encoded, ordinal, number))
        if state == "COMPLETED":
            next_state = "DONE"
        elif state == "FAILED" and failure_count(db, ordinal) > retries:
            next_state = "EXHAUSTED"
        else:
            next_state = "PENDING"
        db.execute("UPDATE jobs SET state=?, receipt=? WHERE ordinal=?", (next_state, encoded, ordinal))


def collect(db, out, ordinal, number, key, retries):
    path = attempt_path(out, ordinal, number) / "receipt.json"
    if not path.is_file():
        return False
    receipt = read_json(path)
    if (receipt["ordinal"], receipt["attempt"], receipt["specification_hash"]) != (ordinal, number, key):
        raise ValueError(f"receipt identity mismatch for case {ordinal}, attempt {number}")
    state = "FAILED" if receipt["failed"] else "COMPLETED"
    finish(db, ordinal, number, state, receipt["reason"], retries, receipt)
    return True


def recover(db, out, retries):
    running = db.execute("""
        SELECT a.ordinal, a.number, j.item FROM attempts a
        JOIN jobs j USING (ordinal) WHERE a.state='RUNNING' ORDER BY a.ordinal
    """).fetchall()
    counts = Counter()
    for ordinal, number, item in running:
        if collect(db, out, ordinal, number, json.loads(item)["specification_hash"], retries):
            counts["recovered_receipts"] += 1
        else:
            finish(db, ordinal, number, "INTERRUPTED", "Previous coordinator ended before a receipt was available.", retries)
            counts["interrupted_attempts"] += 1
    with db:
        for ordinal, in db.execute("SELECT ordinal FROM jobs WHERE state='EXHAUSTED'").fetchall():
            if failure_count(db, ordinal) <= retries:
                db.execute("UPDATE jobs SET state='PENDING' WHERE ordinal=?", (ordinal,))
    return dict(counts)


def begin(db, ordinal):
    with db:
        number = db.execute("SELECT coalesce(max(number),0)+1 FROM attempts WHERE ordinal=?", (ordinal,)).fetchone()[0]
        db.execute("INSERT INTO attempts VALUES (?, ?, 'RUNNING', ?, NULL, NULL, NULL)",
                   (ordinal, number, now()))
        db.execute("UPDATE jobs SET state='RUNNING' WHERE ordinal=?", (ordinal,))
    return number


def schedule(db, out, legacy, workers, retries, stop_after):
    submitted = 0
    while stop_after is None or submitted < stop_after:
        if not db.execute("SELECT 1 FROM jobs WHERE state='PENDING' LIMIT 1").fetchone():
            break
        active = {}
        pool_broken = False
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while True:
                room = workers - len(active)
                if stop_after is not None:
                    room = min(room, stop_after - submitted)
                if not pool_broken and room:
                    pending = db.execute("SELECT ordinal, item FROM jobs WHERE state='PENDING' ORDER BY ordinal LIMIT ?",
                                         (room,)).fetchall()
                    for ordinal, raw in pending:
                        item = json.loads(raw)
                        key = item["specification_hash"]
                        number = begin(db, ordinal)
                        submitted += 1
                        try:
                            future = pool.submit(worker, item, out, ordinal, number, legacy)
                        except BrokenProcessPool as error:
                            finish(db, ordinal, number, "FAILED", str(error), retries)
                            pool_broken = True
                            break
                        active[future] = ordinal, number, key
                if not active:
                    break
                ready, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in ready:
                    ordinal, number, key = active.pop(future)
                    try:
                        future.result()
                        reason = "Worker completed without writing a receipt."
                    except Exception as error:
                        if isinstance(error, BrokenProcessPool):
                            pool_broken = True
                        reason = f"{type(error).__name__}: {error}"
                    if not collect(db, out, ordinal, number, key, retries):
                        finish(db, ordinal, number, "FAILED", reason, retries)
        if not pool_broken:
            break
    return submitted


def export(db, out, reviews, stats):
    rows = []
    for ordinal, raw, repeat, state, receipt in db.execute("SELECT * FROM jobs ORDER BY ordinal"):
        item = json.loads(raw)
        evidence = json.loads(receipt) if receipt else {}
        attempts = [dict(number=number, state=attempt_state, started_at=start, finished_at=end, reason=reason,
                         receipt=f"internal/attempts/{ordinal:04d}/{number:03d}/receipt.json" if written else None)
                    for number, attempt_state, start, end, reason, written in db.execute(
                        "SELECT number, state, started_at, finished_at, reason, receipt FROM attempts "
                        "WHERE ordinal=? ORDER BY number", (ordinal,))]
        prompt = None
        if state == "DONE" and evidence["label_gate"] == "PASS":
            prompt = f"blind/problem-{ordinal:04d}.md"
            path = out / prompt
            rendered = render_problem(item)
            if not path.is_file() or path.read_bytes() != rendered.encode("utf-8"):
                atomic_text(path, rendered)
        observations = reviews.get(item["specification_hash"], [])
        rows.append(dict(**evidence, case_number=ordinal, family=item["family_id"], instance_id=item["instance_id"],
                         parameters=item["parameters"], state=state, duplicate_of=repeat,
                         attempts=attempts, prompt=prompt, blind_status=review_status(observations),
                         blind_observations=observations))
        rows[-1].setdefault("specification_hash", item["specification_hash"])
    report = dict(summary=dict(total=len(rows), states=dict(Counter(row["state"] for row in rows)),
                               reliable=sum(row["prompt"] is not None for row in rows),
                               labels=dict(Counter(row["label"] for row in rows if row["prompt"]))),
                  invocation=stats, rows=rows)
    atomic_text(out / "internal" / "manifest.json", encode(report))
    return report


def run(out, items, reviews, legacy=None, workers=3, retries=1, stop_after=None):
    if workers < 1 or retries < 0 or stop_after is not None and stop_after < 1:
        raise ValueError("workers and stop-after must be positive; retries must be nonnegative")
    if any(item["family_id"] == "A1" and item["validity_checks"]["solver_dispatch"] == "PASS" for item in items):
        if legacy is None:
            raise ValueError("dispatchable A1 cases require --legacy-solver")
        check_release(legacy)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    journal = out / "checkpoint.sqlite3"
    with controller_lock(out):
        if not journal.is_file() and any(path.name != ".controller.lock" for path in out.iterdir()):
            raise ValueError("output directory is nonempty and has no checkpoint")
        with sqlite3.connect(journal) as db:
            db.execute("PRAGMA synchronous=FULL")
            initialize(db, items)
            resumed = db.execute("SELECT count(*) FROM jobs WHERE state='DONE'").fetchone()[0]
            recovered = recover(db, out, retries)
            try:
                submitted = schedule(db, out, legacy, workers, retries, stop_after)
            except KeyboardInterrupt:
                export(db, out, reviews, dict(interrupted=True, previously_completed=resumed, **recovered))
                raise
            return export(db, out, reviews, dict(previously_completed=resumed, attempts_submitted=submitted,
                                                 workers=workers, **recovered))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy-solver", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--stop-after", type=int, help="checkpoint and return after at most this many new attempts")
    parser.add_argument("--reviews", type=Path)
    args = parser.parse_args()
    try:
        report = run(args.out, candidates(args.seed, read_json(args.plan)), load_reviews(args.reviews),
                     args.legacy_solver, args.workers, args.retries, args.stop_after)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({**report["summary"], **report["invocation"]}), flush=True)


if __name__ == "__main__":
    main()
