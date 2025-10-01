# Dockerfile for the Hedging Engine

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in pyproject.toml
RUN pip install --no-cache-dir -r pyproject.toml

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run runner.py when the container launches
CMD ["python", "runner.py"]

# Note: This Dockerfile assumes that there is a pyproject.toml file in the same directory
# as the Dockerfile, and that the main application file is named runner.py. Adjust these
# names as necessary for your project.