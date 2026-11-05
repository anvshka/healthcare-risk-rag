# Start here

The local app lets you pick a patient and admission, inspect the mortality-model score and five SHAP factors, view a separate prolonged-stay ensemble score, and read a cited explanation.

After setting up Python and running the README commands, including the secondary model command below, launch it from this repository folder:

```powershell
.\.venv\Scripts\python.exe -m src.prolonged_stay
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Then open http://127.0.0.1:8501 in your browser. Keep the terminal running; Ctrl+C stops the server.

1. Open the sidebar if it is collapsed (the arrow at the top left).
2. Choose a patient ID, then an admission ID.
3. Read the mortality score, the separate prolonged-stay ensemble score, SHAP chart, factors and explanation. Expand Retrieved sources to inspect the exact passages.
4. Held-out admissions are shown by default. All admissions also includes training patients, which are clearly labeled.

No API key is required. The app does not train a model, download weights or publish patient data when opened. If setup files are missing, it shows the commands needed to prepare them.

**The knowledge files are placeholders, not medical evidence.** Replace them using data/knowledge/README.md to add real literature context. Model scores are retrospective, uncalibrated educational outputs, not clinical advice.

In the original prepared workspace, the companion Open_Clinical_Risk_App.cmd beside the project folder reuses the already installed local environment. That convenience launcher is specific to the prepared workspace; portable setup is documented in README.md.
