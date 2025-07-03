# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies if any (e.g., for specific tools)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package

# Install uv for ultra-fast Python package installation
RUN pip install --no-cache-dir uv

# Install dependencies using uv (much faster than pip)
COPY requirements.txt /usr/src/app/
RUN uv pip install --system --no-cache -r requirements.txt

# Install additional packages for GAIA benchmark testing
RUN uv pip install --system --no-cache datasets

# Copy the .env file
# WARNING: Copying .env directly into the image is not recommended for production.
# Secrets should be managed via runtime environment variables or Docker secrets.
COPY .env /usr/src/app/

# Copy the rest of the application code into the container
COPY ./app /usr/src/app/app
COPY ./config /usr/src/app/config
COPY ./mcp_servers /usr/src/app/mcp_servers
COPY ./tools /usr/src/app/tools

# Copy test files in organized structure
COPY ./tests /usr/src/app/tests

# Copy additional files needed for testing and submission
COPY space_code.py /usr/src/app/
COPY gaia_questions_sample.json /usr/src/app/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application using Uvicorn
# The --reload flag is useful for development but should be removed for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]