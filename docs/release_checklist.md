# Release checklist

Checks before you publish this repo (for example to a new GitHub remote).

## Deliverables

| Item | Location |
|------|----------|
| Scoring CLI | `run.sh` / `run.ps1` |
| Working prototype | `app.py` (Streamlit) + `llm_layer.py` |
| Technical documentation | [`methodology.md`](methodology.md) |
| Architecture overview | [`architecture.md`](architecture.md) |
| Demo workflow | [`demo_walkthrough.md`](demo_walkthrough.md) |

## Automated checks

```bash
# Fresh environment dry run (Linux / Git Bash)
bash scripts/linux_dry_run.sh

# Or manual:
python3 -m venv /tmp/revenue-foresight-venv
source /tmp/revenue-foresight-venv/bin/activate
pip install -r requirements.txt
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv
python src/verify_output.py
python scripts/check_predictions.py   # optional ROAS/spread sanity

# Windows PowerShell
python -m venv .venv_clean
.\.venv_clean\Scripts\Activate.ps1
pip install -r requirements.txt
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
python src/verify_output.py
python scripts/check_predictions.py

# TEST holdout metrics (not required for scoring)
python scripts/evaluate.py
```

## Guide checklist

- [ ] Public GitHub repo
- [ ] `run.sh` at root; works with 3 args + defaults
- [ ] `requirements.txt` fully pinned (`pkg==version`)
- [ ] `data/` present; code reads by pattern (no absolute paths)
- [ ] `pickle/model.pkl` committed and loads under pinned deps
- [ ] Output overwritten at `OUTPUT_PATH` every run
- [ ] No prompts / interactive input
- [ ] No network calls at run time (LLM only in demo)
- [ ] Seeds fixed for training (retrain offline only)
- [ ] Python version stated in README (3.12.x)
- [ ] Docs linked from README
- [ ] `python src/verify_output.py` → `VERIFY OK`

## Publish to a new GitHub repo

Do this from the project root when you are ready (not done automatically):

```powershell
git init
git add run.sh run.ps1 requirements.txt requirements-demo.txt README.md app.py llm_layer.py
git add data/ pickle/ src/ demo/ docs/ scenarios/ scripts/ tests/
git add .gitignore .github
# Do NOT add .venv / .venv_clean / .env / output/*.csv
git status
git commit -m "Initial Revenue Foresight package"
```

Create an empty public repo on GitHub (New repository → Public → do not add a README), then:

```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/revenue-foresight.git
git push -u origin main
```

Or with `gh`:

```powershell
gh auth login
gh repo create revenue-foresight --public --source=. --remote=origin --push
```

### Fresh-environment dry run

```powershell
python -m venv .venv_clean
.\.venv_clean\Scripts\Activate.ps1
pip install -r requirements.txt
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
python src/verify_output.py
```

Must print `VERIFY OK`. Then deactivate and delete `.venv_clean` if you want.

## Scoring mental model

`clone → pip install -r requirements.txt → replace data/ → ./run.sh → read predictions.csv`
