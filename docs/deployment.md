# RetinaScan AI — Cloud Deployment Guide (100% Free Tiers)

This guide details how to deploy **RetinaScan AI** to 100% free cloud platforms for your hackathon presentation.

---

## 1. Which Free Option Should You Choose?

| Goal | Best Platform | Free Tier Specs | Notes |
| :--- | :--- | :--- | :--- |
| **I want our custom HTML/JS UI** (with comparison slider, laser sweep, etc.) | **Render.com** (Native Python) | 512 MB RAM, 100% Free, No credit card | Unified single URL (`onrender.com`). Spins down when idle. |
| **I want 16 GB RAM + Zero sleep** (Fastest ML inference for judges) | **Hugging Face Spaces** (Gradio SDK) | 2 vCPU, 16 GB RAM, 100% Free, No credit card | Uses `app.py` (Gradio UI). Docker Spaces require PRO, but Gradio is 100% free. |

---

## 2. Option 1: Render.com (Hosts Your Custom Web App)

Render allows you to host the unified FastAPI backend + custom frontend for free without using Docker.

### Steps:
1. Push your latest code to your GitHub repo (`main` branch):
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```
2. Go to [Render.com](https://render.com/) and sign in (using GitHub).
3. Click **New + > Web Service**.
4. Select your GitHub repository (`stoic-bruh/retinascan-ai`).
5. Set the following configuration:
   * **Name**: `retinascan-ai`
   * **Language**: `Python 3`
   * **Region**: Any (e.g. Singapore or Frankfurt)
   * **Branch**: `main`
   * **Build Command**:
     ```bash
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
     ```
   * **Start Command**:
     ```bash
     uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
     ```
   * **Instance Type**: `Free`
6. Click **Deploy Web Service**.
7. Render will provide a public URL (e.g., `https://retinascan-ai.onrender.com`). Visiting it opens the full custom UI!

> [!TIP]
> **Hackathon Pro-Tip for Render Free Tier**:
> Render free web services go to sleep after 15 minutes of inactivity. When you are 5 minutes away from presenting to the judges, open your Render link once on your phone or laptop so it's warm and fast when you demo!

---

## 3. Option 2: Hugging Face Spaces (Gradio SDK — 16 GB RAM Free)

Hugging Face Spaces offers 100% free hosting with 16 GB RAM using the **Gradio SDK** (no paid PRO plan needed). We have created [`app.py`](../app.py) for this purpose.

### Steps:
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Settings:
   * **Space name**: `retinascan-ai`
   * **License**: `mit`
   * **Space SDK**: Select **Gradio** (NOT Docker)
   * **Space hardware**: **CPU Basic (2 vCPU · 16 GB RAM · Free)**
   * **Privacy**: **Public**
3. Click **Create Space**.
4. Push your code to the Hugging Face Space git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/retinascan-ai
   git push space main
   ```
5. Hugging Face automatically detects `app.py` and `requirements.txt`, builds the environment, and launches the app at `https://<YOUR_HF_USERNAME>-retinascan-ai.hf.space`.

---

## 4. Option 3: Decoupled (Vercel Frontend + Render API)

If you prefer hosting the custom frontend on a super-fast global CDN like Vercel or Netlify:
1. Deploy your backend to Render as an API.
2. Drag-and-drop the `frontend/` folder to [Netlify Drop](https://app.netlify.com/drop) or import to Vercel.
3. Open your deployed frontend, click the **"API Offline"** status badge in the top-right header, enter your Render API URL (e.g. `https://retinascan-ai.onrender.com`), and click **Save & Connect**.
