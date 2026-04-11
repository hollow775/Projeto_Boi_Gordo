# Review report — split treino até 2024 + holdout 2025 + UI 1..15 dias

## Scope reviewed

Approved spec: `.omx/specs/deep-interview-train-split-2024-ui-15d.md`

Reviewed implementation evidence in the integrated leader branch:

- `main_split_2024_holdout_2025.py`
- `app_split_2024_holdout_2025.py`
- `src/experiments/split_2024_holdout_2025.py`
- `Tests/test_split_2024_holdout_2025.py`
- `Tests/test_train_split_2024_ui_acceptance.py`

## Review summary

The approved flow is implemented as an isolated experiment lane:

- training cutoff fixed at `2024-12-31`
- holdout restricted to `2025-01-01` through `2025-12-31`
- artifacts isolated under `data/processed/train_split_2024_holdout_2025/`
- models isolated under `models_saved/train_split_2024_holdout_2025/`
- Streamlit UI entrypoint available in `app_split_2024_holdout_2025.py`
- 1..15 daily forecast curve composed with the required rule:
  - day 1 -> `h1`
  - days 2..7 -> `h7`
  - days 8..15 -> `h15`

The legacy pipeline remains available through `main.py`.

## Code quality findings

### Verified implementation points

- Temporal split is centralized in `src/experiments/split_2024_holdout_2025.py` via:
  - `TRAIN_END = pd.Timestamp("2024-12-31")`
  - `HOLDOUT_START = pd.Timestamp("2025-01-01")`
  - `HOLDOUT_END = pd.Timestamp("2025-12-31")`
- Training isolation is enforced by `configured_training_runtime(...)`, which redirects:
  - `MODELS_DIR`
  - `DATA_PROCESSED`
  - `CUTOFF_DATE`
- Feature datasets are built once, cached, and reused by both CLI and UI paths.
- UI copy and layout match the requested simple green/orange presentation.
- Manual forecasting keeps fields editable and shows a non-autofill example from the last training day.

### Risks / follow-ups

1. `streamlit` is imported by `app_split_2024_holdout_2025.py`, but no `streamlit` entry was found in `requirements.txt`.
2. `Tests/test_train_split_2024_ui_acceptance.py` still contains heuristic checks that skip two implemented behaviors; stronger coverage currently comes from:
   - `Tests/test_split_2024_holdout_2025.py`
   - direct import/help smoke checks
3. `build_feature_row_for_manual_inputs(...)` triggers `pandas` `PerformanceWarning` messages through `src/features/engineering.py`; this is not a functional failure, but it is a maintainability/performance smell worth future cleanup.

## Verification evidence

### CLI entrypoint

Command:

```powershell
python main_split_2024_holdout_2025.py --help
```

Result:

- PASS — help output exposes `--train`, `--evaluate`, `--full`, and `--no-cache`.

### Python syntax / import sanity

Command:

```powershell
python -m py_compile `
  main_split_2024_holdout_2025.py `
  app_split_2024_holdout_2025.py `
  src\experiments\split_2024_holdout_2025.py `
  Tests\test_split_2024_holdout_2025.py `
  Tests\test_train_split_2024_ui_acceptance.py
```

Result:

- PASS

### UI dependency/import smoke check

Command:

```powershell
python -c "import streamlit, app_split_2024_holdout_2025; print('streamlit_import_ok')"
```

Result:

- PASS — printed `streamlit_import_ok`

### Focused test suite

Command:

```powershell
python -m unittest Tests.test_split_2024_holdout_2025 Tests.test_train_split_2024_ui_acceptance -v
```

Result:

- PASS — `Ran 7 tests ... OK (skipped=2)`
- Confirmed behavior:
  - cutoff/holdout separation checks pass
  - composed 1..15 curve rule passes in `Tests/test_split_2024_holdout_2025.py`
  - manual input feature-row generation passes
- Note:
  - the 2 skipped cases come from heuristic acceptance checks that have not yet been refreshed to assert the integrated implementation directly

### Static diagnostics

Current project state reviewed from the Python repo:

- no formal lint configuration was found
- no Python typecheck configuration was found
- `lsp_diagnostics` returned no errors for the review-test file used in this lane

## Operator runbook

### Generate the isolated 2024/2025 flow

```powershell
python main_split_2024_holdout_2025.py --full
```

### Train only

```powershell
python main_split_2024_holdout_2025.py --train
```

### Evaluate only

```powershell
python main_split_2024_holdout_2025.py --evaluate
```

### Rebuild caches

```powershell
python main_split_2024_holdout_2025.py --full --no-cache
```

### Launch the UI

```powershell
streamlit run app_split_2024_holdout_2025.py
```

## Expected artifact locations

- Processed outputs:
  - `data/processed/train_split_2024_holdout_2025/`
- Holdout CSVs/plots:
  - `data/processed/train_split_2024_holdout_2025/holdout_2025/`
- Experiment models:
  - `models_saved/train_split_2024_holdout_2025/`
- UI helper artifacts:
  - `data/processed/train_split_2024_holdout_2025/exemplo_ultimo_dia_treino.csv`
  - `data/processed/train_split_2024_holdout_2025/historico_recente_boi_gordo.csv`

## Final assessment

The requested 2024/2025 split flow and UI are present and structured cleanly as a separate experiment path. The main remaining review concerns are reproducible dependency declaration for Streamlit and refreshing the older heuristic acceptance checks so they validate the landed implementation directly.
