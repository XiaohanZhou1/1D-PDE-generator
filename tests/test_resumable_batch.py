from collections import Counter
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from pdegen.batch_workflow import candidates
from pdegen.resumable_batch import atomic_text, attempt_path, begin, encode, initialize, now, recover, schedule


def transient_worker(item, out, ordinal, number, legacy):
    receipt = dict(ordinal=ordinal, attempt=number, specification_hash=item["specification_hash"],
                   finished_at=now(), failed=number == 1,
                   reason="deliberate transient failure" if number == 1 else "completed")
    atomic_text(attempt_path(out, ordinal, number) / "receipt.json", encode(receipt))


class ResumableBatchTests(unittest.TestCase):
    def setUp(self):
        self.items = candidates(1, [{"family": "E3", "parameters": {"gain": gain}} for gain in (5, 8)])

    def test_retry_retains_failed_attempt_and_stops_after_budget(self):
        with TemporaryDirectory() as directory, sqlite3.connect(":memory:") as db:
            out = Path(directory)
            initialize(db, self.items[:1])
            with patch("pdegen.resumable_batch.worker", transient_worker):
                self.assertEqual(schedule(db, out, None, 1, 1, 1), 1)
                self.assertEqual(db.execute("SELECT state FROM jobs").fetchone()[0], "PENDING")
                self.assertEqual(schedule(db, out, None, 1, 1, None), 1)
                self.assertEqual(schedule(db, out, None, 1, 1, None), 0)
            rows = db.execute("SELECT state, reason FROM attempts ORDER BY number").fetchall()
            self.assertEqual(rows, [("FAILED", "deliberate transient failure"), ("COMPLETED", "completed")])

    def test_recover_uncommitted_receipt_and_interrupted_attempt(self):
        with TemporaryDirectory() as directory, sqlite3.connect(":memory:") as db:
            out = Path(directory)
            initialize(db, self.items)
            first = begin(db, 1)
            begin(db, 2)
            receipt = dict(ordinal=1, attempt=first, specification_hash=self.items[0]["specification_hash"],
                           finished_at=now(), failed=False, reason="completed")
            atomic_text(attempt_path(out, 1, first) / "receipt.json", encode(receipt))
            self.assertEqual(recover(db, out, 0), {"recovered_receipts": 1, "interrupted_attempts": 1})
            self.assertEqual(Counter(row[0] for row in db.execute("SELECT state FROM jobs")),
                             {"DONE": 1, "PENDING": 1})
            self.assertEqual(begin(db, 2), 2)
            with self.assertRaisesRegex(ValueError, "plan or seed"):
                initialize(db, self.items[::-1])
