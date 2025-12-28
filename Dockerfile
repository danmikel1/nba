# Use a lightweight Python version
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirement file first (for caching)
COPY requirements.txt .

# Install dependencies
# We also install 'curl' for healthchecks if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . .

# Expose Streamlit's default port
EXPOSE 9625

# Run the app
CMD ["streamlit", "run", "nbav13_refactored.py", "--server.port=9625", "--server.address=0.0.0.0"]