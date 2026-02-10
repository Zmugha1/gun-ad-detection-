# Deploy to GitHub and run on Streamlit Cloud

## Push this repo to GitHub

1. **Create the repo** on GitHub: [https://github.com/Zmugha1/gun-ad-detection-](https://github.com/Zmugha1/gun-ad-detection-) (already exists; it's empty).

2. **From the `gun-ad-detection` folder** (this project root):

```powershell
cd "c:\Users\zumah\OneDrive\Desktop\DrMughalsResearchIntelligenceStudio\MLDLC 2.0\gun-ad-detection"

# Initialize git if not already
git init
git add .
git commit -m "Weapons Detection Content Moderation: 3 approaches, Streamlit app, calibration and cost analysis"

# Add your GitHub remote (use the repo URL)
git remote add origin https://github.com/Zmugha1/gun-ad-detection-.git

# Push (main branch)
git branch -M main
git push -u origin main
```

If the repo already had a remote and content, use:

```powershell
git remote add origin https://github.com/Zmugha1/gun-ad-detection-.git
git push -u origin main
```

## Run locally

```powershell
cd gun-ad-detection
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run_app.bat` after creating the venv and installing requirements.

## Run on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with GitHub and select the repo `Zmugha1/gun-ad-detection-`.
3. **Main file path:** `app.py`
4. **Branch:** `main`
5. Click **Deploy**. Streamlit will install from `requirements.txt` and run `streamlit run app.py`.

**Note:** The app uses a keyword-based fallback if `transformers`/`torch` are not installed or loading BERT fails (e.g. on free-tier memory). For full BERT inference, use a machine with enough RAM or a paid Streamlit plan.
