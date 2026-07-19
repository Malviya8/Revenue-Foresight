# Submission email template

**To:** sunitha.k@netelixir.us  
**Subject:** AIgnition 3.0 Submission — `<TEAM_NAME>`  
**Deadline:** 19 July 2026, 10:00 PM IST  

Fill `<TEAM_NAME>`, members, and college before sending. Repo URL is ready.

---

Hello NetElixir team,

Please find our AIgnition 3.0 submission below.

**Team name:** `<TEAM_NAME>`  
**Members:** `<NAME_1>`, `<NAME_2>`, …  
**College:** `<COLLEGE>`  

**Public GitHub repository:**  
https://github.com/Malviya8/aignition-forecasting  

**Exact scoring command:**  
```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

**Python:** 3.12.x  
**Dependencies:** pinned in `requirements.txt`  
**Trained artifact:** `pickle/model.pkl` (committed; scorers do not retrain)

**Holdout (sample data, TEST):** wMAPE 0.500 vs run-rate 1.234 (59.5% lift, excl. Bing); P10–P90 coverage 74.0%.  
See `docs/backtest_results.md` / `docs/evaluation_snapshot.json`.

**Documentation (in repo):**
- Technical methodology — `docs/methodology.md`
- Architecture — `docs/architecture.md`
- Demo walkthrough — `docs/demo_walkthrough.md`
- Submission checklist — `docs/submission_checklist.md`

**Demo (optional for evaluation / presentation):**
```bash
pip install -r requirements-demo.txt
streamlit run app.py
```

Thank you,  
`<TEAM_NAME>`

---

## Pre-send checklist

- [ ] Repo is **public**
- [ ] `pickle/model.pkl` is committed and pullable without Git LFS issues
- [ ] Fresh clone dry-run: install → `./run.sh` → valid `predictions.csv`
- [ ] `python src/verify_submission.py` → `VERIFY OK`
- [ ] Team / college / email fields filled (`<TEAM_NAME>`, members, college)
