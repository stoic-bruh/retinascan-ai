# RetinaScan AI — Cloud Deployment Guide (100% Free Tiers)

This guide details how to deploy **RetinaScan AI** to 100% free cloud platforms for your hackathon presentation.

---

## 1. Cloud Deployment: Render.com (Hosts Web App & API)

Render hosts the unified FastAPI backend + custom frontend on a single URL:
**https://retinascan-ai-73ss.onrender.com**

### Render Configuration Used:
* **Environment**: `Python 3`
* **Build Command**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
  ```
* **Start Command**:
  ```bash
  uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1
  ```
* **Instance Type**: `Free`

> [!TIP]
> **Hackathon Presentation Tip for Render Free Tier**:
> Render free web services spin down after 15 minutes of inactivity. Open `https://retinascan-ai-73ss.onrender.com` 2–3 minutes before presenting to the judges to wake the container so scans are instant.

## 2. Alternative: Decoupled (Static Frontend CDN + Render API)

If you prefer hosting the custom frontend on a static CDN like Vercel or Netlify:
1. Deploy your backend to Render as an API.
2. Drag-and-drop the `frontend/` folder to [Netlify Drop](https://app.netlify.com/drop) or import to Vercel.
3. Open your deployed frontend, click the connection status badge in the top-right header, enter your Render API URL (`https://retinascan-ai-73ss.onrender.com`), and click **Save & Connect**.
