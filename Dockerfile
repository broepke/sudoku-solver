# Use an official Python base image
FROM python:3.9-slim

# Install system dependencies for Tesseract and OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy your application code
COPY . /app
WORKDIR /app

# Expose ports for Streamlit
EXPOSE 8501

# Run the Streamlit app with correct host binding
CMD ["streamlit", "run", "Sudoku_Solver.py", "--server.address=0.0.0.0"]