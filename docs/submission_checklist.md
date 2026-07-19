# Submission checklist

Before emailing `sunitha.k@netelixir.us` (**19 Jul 2026, 10:00 PM IST**):

## Deliverables map (brief)

| Brief item | Location |
|------------|----------|
| Working prototype | `run.sh` + `app.py` (Streamlit) + `llm_layer.py` |
| Technical documentation | [`methodology.md`](methodology.md) |
| Architecture overview | [`architecture.md`](architecture.md) |
| Demo workflow | [`demo_walkthrough.md`](demo_walkthrough.md) |

Email draft: [`SUBMISSION_EMAIL.md`](SUBMISSION_EMAIL.md).

## Automated checks

```bash
# Fresh environment dry run (Linux / Git Bash)
bash scripts/linux_dry_run.sh

# Or manual:
python3 -m venv /tmp/aignition-venv
source /tmp/aignition-venv/bin/activate
pip install -r requirements.txt
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv
python src/verify_submission.py

# Windows PowerShell
python -m venv .venv_clean
.\.venv_clean\Scripts\Activate.ps1
pip install -r requirements.txt
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
python src/verify_submission.py
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
- [ ] `python src/verify_submission.py` → `VERIFY OK`
- [ ] Email draft filled (`docs/SUBMISSION_EMAIL.md`)

## From-zero packaging (do this once)

### A. Install Git + GitHub CLI (if needed)

1. Install [Git for Windows](https://git-scm.com/download/win)
2. Create a free account at [github.com](https://github.com)
3. Optional but handy: install [GitHub CLI](https://cli.github.com/) (`gh`)

### B. Turn this folder into a public repo

In PowerShell from the project root:

```powershell
git init
git add run.sh run.ps1 requirements.txt requirements-demo.txt README.md app.py llm_layer.py
git add data/ pickle/ src/ demo/ docs/ scenarios/ scripts/
git add .gitignore
# Do NOT add .venv / .venv_clean / .env / output/*.csv
git status
git commit -m "Initial AIgnition Forecasting submission package"
```

Create the empty public repo on GitHub (website: New repository → Public → do not add README), then:

```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin main
```

Or with `gh`:

```powershell
gh auth login
gh repo create <YOUR_REPO> --public --source=. --remote=origin --push
```

### C. Fresh-environment dry run (mandatory)

```powershell
python -m venv .venv_clean
.\.venv_clean\Scripts\Activate.ps1
pip install -r requirements.txt
.\run.ps1 ./data ./pickle/model.pkl ./output/predictions.csv
python src/verify_submission.py
```

Must print `VERIFY OK`. Then deactivate and delete `.venv_clean` if you want.

### D. Email organizers

Fill `docs/SUBMISSION_EMAIL.md` and send to `sunitha.k@netelixir.us` by **19 Jul 2026, 10:00 PM IST** with:
1. Public GitHub URL  
2. Exact command `./run.sh ./data ./pickle/model.pkl ./output/predictions.csv`  
3. Team name, members, college  

## Scorer mental model

`clone → pip install -r requirements.txt → replace data/ → ./run.sh → read predictions.csv`
