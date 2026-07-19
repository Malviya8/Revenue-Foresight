# AIgnition Forecasting — Demo

```bash
pip install -r requirements-demo.txt
streamlit run app.py
```

**Free LLM (recommended):** create a key at https://console.groq.com then:

```powershell
$env:GROQ_API_KEY="gsk_..."
streamlit run app.py
```

Or paste the Groq key in the sidebar (provider = `auto` / `groq`).  
No key → offline heuristic briefing. OpenAI still works via `OPENAI_API_KEY`.

Files: `app.py`, `llm_layer.py` (not used by `run.sh`).
