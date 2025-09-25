
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_enhanced.txt .
RUN pip install --no-cache-dir -r requirements_enhanced.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
