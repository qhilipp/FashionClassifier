# Base-Image with Python 3.13
FROM python:3.13

# Work directory in the container
WORKDIR /app

# Install all dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Start command for the Python script
CMD ["python", "model.py"]
