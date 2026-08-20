FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (src/ layout - see pyproject.toml) and register the
# local package. --no-deps: dependencies are already installed above from the
# pinned requirements.txt; this step only makes `lahmajo` importable.
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Create necessary directories
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/documents || exit 1

# Run the application
CMD ["uvicorn", "lahmajo.api.routes:app", "--host", "0.0.0.0", "--port", "8080"]