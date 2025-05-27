# Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Ensure entrypoint.sh is executable
RUN chmod +x /app/entrypoint.sh

# Default entrypoint (runs full test pipeline incl. A/B test)
ENTRYPOINT ["/app/entrypoint.sh"]
