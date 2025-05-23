# Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .
COPY Data/ Data/

# Run all tests on container start
CMD ["python", "run_all_tests.py"]
