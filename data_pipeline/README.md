# Data Pipeline

## Blocking Asset Checks

Each check uses `@asset_check(blocking=True)` — when a check fails, Dagster halts materialization of all downstream assets.

| Check | Asset | What it catches |
|-------|-------|----------------|
| `raw_dataset_is_nonempty` | `raw_dataset` | Download failure or over-sampling producing 0 rows |
| `raw_dataset_has_text_column` | `raw_dataset` | Schema drift in upstream HuggingFace dataset |
| `cleaned_dataset_is_nonempty` | `cleaned_dataset` | Overly aggressive cleaning removing all rows |
| `cleaned_dataset_no_empty_texts` | `cleaned_dataset` | Empty strings surviving the cleaning filter |
| `tokenized_dataset_has_expected_columns` | `tokenized_dataset` | Missing `input_ids` or `attention_mask` after tokenization |
| `tokenized_dataset_is_nonempty` | `tokenized_dataset` | Tokenization producing empty dataset |
| `tokenized_dataset_sequences_valid` | `tokenized_dataset` | All-zero token sequences (corrupted tokenization) |
| `train_val_splits_exist_on_disk` | `train_val_splits` | Missing or empty train/val directories on disk |

No check exists for `trained_model` since it is the terminal asset with nothing downstream to protect.

## Resilient, Atomic State Machine

The pipeline is designed as a chain of **atomic state functions**. Each asset is an independent, self-contained unit of work: it reads its inputs, produces an output, and persists that output to disk. No asset holds in-memory state that another asset depends on — all communication happens through materialized artifacts.

This gives the pipeline two key properties:

### Atomicity

Each asset either fully completes and persists its output, or it doesn't. There is no partial state. If `tokenized_dataset` fails halfway through, its previous successful materialization remains intact on disk — downstream assets never see a half-written result. The blocking checks reinforce this: they validate the *persisted* output before any downstream asset is allowed to run, ensuring that every state transition in the pipeline is from one known-good state to another.

### Fault Tolerance and Replay

Because each asset's output is durably persisted, the pipeline is resilient to interruptions at any point:

- **Crash recovery**: If a run is killed mid-pipeline (machine restart, OOM, timeout), you resume by materializing only the assets that haven't completed yet. A 2-hour tokenization job that finished before the crash is not re-run.
- **Selective replay**: Change a config value (e.g., `max_seq_length`) and rematerialize from `tokenized_dataset` onward — `raw_dataset` and `cleaned_dataset` are untouched. Each asset is an independent checkpoint you can roll forward from.
- **Safe re-entry**: Blocking checks run on every materialization, including replays. If you resume from `cleaned_dataset` after a crash and the persisted data turns out to be corrupt, the check catches it before `tokenized_dataset` starts — you never silently propagate bad state.

```
raw_dataset ──check──▸ cleaned_dataset ──check──▸ tokenized_dataset ──check──▸ train_val_splits ──check──▸ trained_model
     ↓                      ↓                          ↓                           ↓
  [persisted]            [persisted]                [persisted]                 [persisted]
```

Each arrow is a **guarded state transition**: the blocking check acts as a gate between the persisted output and the next computation.

### Tradeoffs

- **Disk cost**: Every intermediate asset is persisted to disk. For large datasets this can multiply storage usage by ~4x (raw + cleaned + tokenized + splits). This is a deliberate tradeoff — we favor recoverability over storage efficiency.
- **Check overhead**: Blocking checks add latency to each materialization step. For this pipeline the checks are lightweight (row counts, column presence), so the overhead is negligible relative to the asset computation itself.
- **No rollback**: Atomicity here is forward-only. If `cleaned_dataset` materializes with a subtle bug that passes checks, downstream assets built on it must be manually rematerialized. The checks reduce this risk but don't eliminate it.

### Where to find them

- Check implementations: [`checks.py`](./checks.py)
- Tests: [`../tests/test_checks.py`](../tests/test_checks.py)
- Registration: [`definitions.py`](./definitions.py) via `load_asset_checks_from_modules`
