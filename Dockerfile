# Use the official Azure Functions Python image
FROM mcr.microsoft.com/azure-functions/python:4-python3.10

# Required for Azure Functions to find your code
ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

# Install system-level dependencies (if needed for pandas/numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the entire project to the container
COPY . /home/site/wwwroot

WORKDIR /home/site/wwwroot

ENV PYTHONPATH=/home/site/wwwroot

