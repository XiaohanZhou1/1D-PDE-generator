# Resumable process-isolated batch

The production runner uses the same canonical constructors, proof screens, label acceptance and blind prompt renderer as the unified serial workflow. Worker processes solve one case at a time in separate attempt directories. The coordinator is the only process that writes the SQLite journal.

## Run and resume

Use the normal project dependencies and the pinned original A1 solver release when the plan includes dispatchable A1 cases. Supply the same ordered plan and seed on resume. Choose a new output folder to begin a different batch.

~~~powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -m pdegen.resumable_batch --plan examples/public-batch-plan.json --workers 3 --retries 1 --stop-after 3 --out scratch/public-resume
python -m pdegen.resumable_batch --plan examples/public-batch-plan.json --workers 3 --retries 1 --out scratch/public-resume
~~~

The first example stops after at most three new attempts and checkpoints any that were submitted. The second resumes from the same journal. A process-level lock prevents multiple coordinators changing the same queue. Completed work stays complete. An atomically written worker receipt is recovered if the coordinator stopped before committing it. A started attempt without a receipt is recorded as interrupted and placed back in the queue. Any previous worker that finishes late writes to its old attempt directory.

An execution failure is recorded with its reason and can use the configured retry budget. An interrupted attempt preserves the failure budget. A completed scientific `UNCERTAIN` result is retained for review. The runner makes no automatic claim about the stability of that case. Repeated specifications in a plan link to the first case. A plan or seed change requires a new run folder.

## Outputs

- `checkpoint.sqlite3` is the durable journal of jobs and attempts. Copy this file and its run directory together for a resumed run.
- `internal/attempts/` keeps each attempt separate. A completed attempt has a receipt with its process ID, timings, audit and result paths. Raw A1 artifacts are stored in that attempt.
- `internal/manifest.json` summarizes the current state and includes each case's ordered attempt history with timestamps and reasons. The journal remains authoritative after an abrupt stop before a fresh manifest was exported.
- `blind/` contains the reliably labelled prompts. It is suitable for sharing separately from the internal artifacts.

## A–F interruption demonstration

The 18-case demonstration uses three previously unseen specifications per family. The first coordinator ran with three workers and was terminated during the batch. At interruption six cases had committed, three were running and nine were pending. The next invocation used the same journal. It recovered one durable uncommitted receipt, recorded two interrupted attempts and submitted eleven new attempts. No completed case acquired a new attempt, and all six pre-interruption evidence hashes were unchanged. The finished run had 18 completed cases, 14 reliable labels and four uncertain results.

The retry regression uses a subprocess that intentionally fails its first attempt and succeeds on the next. It checks that both attempt outcomes survive and a completed job is not queued again.
