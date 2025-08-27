# Use official Python image as base
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy everything into /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir pandas scikit-learn matplotlib seaborn dash numpy

# Default command (run app.py inside /app/app)
CMD ["python", "app/app.py"]
