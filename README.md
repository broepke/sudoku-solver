# Sudoku Solver with OCR

This project is a Sudoku solver application built using Streamlit and Tesseract OCR. It allows users to upload Sudoku puzzle images, extract the grid using Tesseract OCR, and solve the puzzle interactively.

## Features
- Upload a Sudoku puzzle image.
- Automatically extract and display the Sudoku grid.
- Solve the puzzle using a backtracking algorithm.
- Option to manually adjust extracted numbers.
- Built with Streamlit for an interactive user interface.

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

2.	Create a virtual environment (optional but recommended):

```
python3 -m venv venv
source venv/bin/activate
```

3.	Install the required Python packages:
```
pip install -r requirements.txt
```

### 2. Install Tesseract OCR
```
brew install tesseract
```

### 3. Run the Application

1.	Start the Streamlit app:

```
streamlit run sudoku_solver.py
```

### Optional: Docker Deployment

You can also use Docker to run the application with Tesseract pre-installed. A Dockerfile is provided in the repository.

To build and run the Docker container:

```
docker build -t sudoku-solver .
docker run -p 8501:8501 sudoku-solver
```

### Example Usage

1.	Upload Image:
   - Drag and drop a clear Sudoku puzzle image.
2.	Extract Sudoku Grid:
   - The app will preprocess the image and extract the grid.
3.	Solve Sudoku:
   - Click the “Solve” button to display the solution.
4.	Adjust Numbers (Optional):
   - Use the manual grid editor if OCR misses or misinterprets a digit.
