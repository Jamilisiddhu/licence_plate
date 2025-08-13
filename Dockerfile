# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- ADDED STEP: Pre-download EasyOCR models during the build process ---
# This is the key to solving the memory issue.
# We'll run a Python command to download the models directly.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application using gunicorn, a production-ready WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
