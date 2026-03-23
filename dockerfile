# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create the templates directory if not existing
RUN mkdir -p templates

# Make port available to the world outside this container
EXPOSE 8080

# Run the application when the container launches
CMD ["python", "app.py"]
