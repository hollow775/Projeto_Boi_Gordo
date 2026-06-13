# Production lane: local-first daily update

This is the planned non-cloud automation path for the production model. In this first pass, MySQL stores production source snapshots, run audit metadata, CEPEA provenance, and model-version metadata; the training/export dataset is still rebuilt through the existing collector/merger stack after the production refresh succeeds. A later hardening step can make MySQL the sole downstream source of truth for training/export reads.

## What stays separate

- **TCC/academic lane:** keeps training through 2024 and validating on 2025.
- **Production lane:** uses all available data, writes to production-scoped folders, and can evolve daily:
  - `models_saved/production/`
  - `data/processed/production/`
  - `data/outputs/production/`
  - `data/production_last_run.json`

## Data update policy

Daily API-backed collectors:

- BCB PTAX
- ComexStat
- IBGE SIDRA
- Copernicus
- Base deflacionária / inflation index

CEPEA is intentionally manual for this first production pass. Put the downloaded CEPEA files in `data/raw/` using the configured names in `config/settings.py`. Scraping is a later, separate improvement.

## Required MySQL environment variables

Set these before running the production command:

```powershell
$env:BOI_DB_HOST = "localhost"
$env:BOI_DB_PORT = "3306"
$env:BOI_DB_NAME = "boi_gordo"
$env:BOI_DB_USER = "boi_user"
$env:BOI_DB_PASSWORD = "your_password"
```

## Manual first run

From the repository root:

```powershell
.\.venv\Scripts\python.exe production_daily.py --init-db --start-date 2010-01-01
```

For a normal daily run:

```powershell
.\.venv\Scripts\python.exe production_daily.py
```

The command refreshes API-backed production source snapshots into MySQL, records audit/provenance metadata, rebuilds the current training/export dataset through the existing collector/merger stack, trains/version production models, exports website CSVs, and writes `data/production_last_run.json`.

If any required API collector fails, the command aborts before retraining/exporting so the website is not updated with partial or stale data under a successful scheduler status.
During the run, the configured manual CEPEA files are also hashed and recorded in MySQL `manual_source_files` for provenance.

### First-pass architecture boundary

This runbook intentionally documents the current local-first boundary rather than claiming a fully database-sourced production pipeline:

- MySQL is the production persistence and audit layer for refreshed API source snapshots, collector run status, manual CEPEA file provenance, and production model-version metadata.
- Training and website export still rebuild their feature inputs through the existing project collector/merger path after the production refresh succeeds.
- A successful scheduler run therefore proves the local production command completed and wrote production artifacts, but it does not yet prove that every downstream training/export row was read back from MySQL.
- If stricter reproducibility is required later, add a production read/query layer in `src/production/db.py` and make `production_daily.py` build features from those persisted production tables before retraining/exporting.

## Windows Task Scheduler setup

The scheduler should call the same local-first command used manually; it should not call Streamlit and it should not write to the TCC/academic output folders. Use the repository root as the working directory so relative paths such as `data/raw/`, `data/outputs/production/`, and `.venv/` resolve correctly.

Create a daily task with:

- **Program/script:** `C:\Projetos\TrabalhoCurso\BoiGordo\Projeto_Boi_Gordo\.venv\Scripts\python.exe`
- **Arguments:** `production_daily.py`
- **Start in:** `C:\Projetos\TrabalhoCurso\BoiGordo\Projeto_Boi_Gordo`

Recommended schedule settings:

- Run once per day after the manual CEPEA files have been updated, or use `--skip-refresh` / `--skip-train` only for an intentional maintenance run.
- Enable **Run whether user is logged on or not** only after confirming the configured Windows account can read the project folder, `.venv`, CEPEA files, and MySQL credentials.
- Enable **Stop the task if it runs longer than** a conservative limit such as 2 hours.
- Keep **Start in** populated; leaving it blank may make the command run from `C:\Windows\System32` and fail to find project-relative files.

The task must run under a Windows user that has:

1. MySQL access through the `BOI_DB_*` environment variables.
2. CEPEA files already updated manually in `data/raw/`.
3. Copernicus credentials configured if Copernicus refresh is enabled.

### Persisting environment variables for the scheduler user

Task Scheduler does not inherit temporary PowerShell `$env:` values from an interactive terminal. Configure the variables for the same Windows user that will run the task, then open a new terminal or restart the scheduled task service before testing:

```powershell
setx BOI_DB_HOST "localhost"
setx BOI_DB_PORT "3306"
setx BOI_DB_NAME "boi_gordo"
setx BOI_DB_USER "boi_user"
setx BOI_DB_PASSWORD "your_password"
```

Use Windows Credential Manager, a protected user profile, or another secrets mechanism later if this moves beyond the local-first setup. Do not commit real database passwords to the repository.

### Optional scheduled command with logging

For easier troubleshooting, schedule PowerShell instead of `python.exe` and redirect output to a local log file:

- **Program/script:** `powershell.exe`
- **Arguments:** `-NoProfile -ExecutionPolicy Bypass -Command "Set-Location 'C:\Projetos\TrabalhoCurso\BoiGordo\Projeto_Boi_Gordo'; .\.venv\Scripts\python.exe production_daily.py *> logs\production_daily_task.log"`
- **Start in:** `C:\Projetos\TrabalhoCurso\BoiGordo\Projeto_Boi_Gordo`

Create `logs/` locally if it does not exist. The repository does not need to track scheduler logs.

### Optional `schtasks` example

This creates a daily 07:00 task for the current Windows user. Adjust the path and time to match the local machine:

```powershell
schtasks /Create /TN "BoiGordoProductionDaily" /SC DAILY /ST 07:00 /F /TR "\"C:\Projetos\TrabalhoCurso\BoiGordo\Projeto_Boi_Gordo\.venv\Scripts\python.exe\" production_daily.py"
```

After creating the task, open its properties and set **Start in** to `C:\Projetos\TrabalhoCurso\BoiGordo\Projeto_Boi_Gordo`, because `schtasks /Create` does not expose the working-directory field cleanly.

### Verification checklist

Before relying on the daily task:

1. Run `production_daily.py` manually from the repository root with the same Windows user.
2. Run the scheduled task once with **Run** in Task Scheduler.
3. Confirm the task exit code is `0x0`.
4. Confirm `data/production_last_run.json` has a new timestamp.
5. Confirm `data/outputs/production/predictions.csv` and `data/outputs/production/price_history.csv` were updated.
6. If the run fails, inspect `logs/production_daily_task.log` when using the PowerShell logging form, then check MySQL credentials, CEPEA files, and Copernicus credentials.

## Website output

The Streamlit app now prefers production outputs from `data/outputs/production/`:

- `predictions.csv`
- `price_history.csv`

If those files are missing, it falls back to the legacy `data/outputs/` files.
