FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install CPU PyTorch first to keep image lightweight (~700MB vs 4GB+ CUDA wheels)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model checkpoint
COPY backend/ ./backend/
COPY model/ ./model/
COPY frontend/ ./frontend/

# Expose port (7860 default for Hugging Face Spaces, overridable via $PORT)
EXPOSE 7860

# Launch Uvicorn server binding to 0.0.0.0 and dynamic port
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
