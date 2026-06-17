# Python 3.12 slim — matches the project's tested environment
FROM python:3.12-slim

# libgomp1 = OpenMP runtime for Linux (this is the libomp.dylib equivalent
# that was causing the Mac error — Docker sidesteps that entirely)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure data/models dirs exist even if empty (mounted as volumes anyway)
RUN mkdir -p data/raw data/processed models

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
