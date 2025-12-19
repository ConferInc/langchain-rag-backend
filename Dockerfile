
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY rag_system.py .

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Environment variables (set these in Coolify)
# OPENAI_API_KEY - Your OpenAI API key
# OPENAI_API_BASE - OpenAI API base URL (optional)
# QDRANT_URL - Qdrant database URL
# QDRANT_API_KEY - Qdrant API key

# Run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
