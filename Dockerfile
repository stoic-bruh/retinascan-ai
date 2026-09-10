FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

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

# Expose port (default 8000, overridable via $PORT on cloud hosts like Render)
EXPOSE 8000

# Launch Uvicorn server binding to 0.0.0.0 and dynamic port
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
