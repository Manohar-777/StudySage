FROM python:3.11-slim

# Set environment variables to keep Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies (optional: uncomment poppler & tesseract if planning to build OCR inside Docker)
# RUN apt-get update && apt-get install -y \
#     poppler-utils \
#     tesseract-ocr \
#     tesseract-ocr-tel \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
