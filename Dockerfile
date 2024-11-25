# Use an official Python base image
FROM python:3.9-slim

# Install Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy your application code
COPY . /app
WORKDIR /app

# Run the Streamlit app
CMD ["streamlit", "run", "Sudoku_Solver.py"]