# Sudoku Solver with OCR

This project is a Sudoku solver application built using Streamlit and Tesseract OCR. It allows users to upload Sudoku puzzle images, extract the grid using Tesseract OCR, and solve the puzzle interactively.

## Features
- Upload a Sudoku puzzle image.
- Automatically extract and display the Sudoku grid.
- Solve the puzzle using a backtracking algorithm.
- Option to manually adjust extracted numbers.
- Built with Streamlit for an interactive user interface.

![Sudoku Grid](sudoku_easy.png) 

## Installation Instructions

Follow these steps to set up the Sudoku Solver on your local machine.

### Prerequisites
1. **Python**: Ensure you have Python 3.9+ installed.
2. **Tesseract OCR**: The application relies on Tesseract for OCR.

### 1. Install Python and Required Libraries

1. Clone the repository:
```
git clone https://github.com/broepke/sudoku-solver
cd sudoku-solver
```

2.	Run all these commands to crete a virtual environment and install the requirements.

```
python -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt --upgrade
```

### 2. Install Tesseract OCR
```
brew install tesseract
```

## Optional: Docker Deployment

You can also use Docker to run the application with Tesseract pre-installed. A Dockerfile is provided in the repository.

To build and run the Docker container:

```
docker build -t sudoku-solver .
docker run -p 8501:8501 sudoku-solver
```

## Optional: Streamlit Community Cloud

You can also us the Streamlit Community Cloud.  The `packages.txt` file contains the necessary Linux dependencies to but able to support the OCR Package.

## Run the Application

1.	Start the Streamlit app:

```
streamlit run sudoku_solver.py
```

## Example Usage

1.	Upload Image:
   - Drag and drop a clear Sudoku puzzle image.
2.	Extract Sudoku Grid:
   - The app will preprocess the image and extract the grid.
3.	Solve Sudoku:
   - Click the “Solve” button to display the solution.
4.	Adjust Numbers (Optional):
   - Use the manual grid editor if OCR misses or misinterprets a digit.
