import streamlit as st
import numpy as np
import cv2
import pytesseract
import tempfile
import platform
import logging

# Configure the logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Log the information
logging.info("OpenCV Version: %s", cv2.__version__)
logging.info("Pytesseract Version: %s", pytesseract.get_tesseract_version())
logging.info("Python Version: %s", platform.python_version())


def preprocess_image(image):
    """Preprocess the image for better OCR performance."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Remove gridlines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    vertical_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    grid_lines = cv2.add(horizontal_lines, vertical_lines)
    grid_removed = cv2.subtract(thresh, grid_lines)

    return grid_removed


def extract_sudoku_grid_cells(processed_image):
    """Extract each cell of the Sudoku grid and recognize the digits using OCR."""
    grid_size = processed_image.shape[0]
    cell_size = grid_size // 9  # Each cell is approximately 1/9th of the grid
    sudoku_grid = np.zeros((9, 9), dtype=int)

    for row in range(9):
        for col in range(9):
            # Extract each cell
            x_start, y_start = col * cell_size, row * cell_size
            x_end, y_end = x_start + cell_size, y_start + cell_size
            cell = processed_image[y_start:y_end, x_start:x_end]

            # Add padding for OCR
            cell = cv2.copyMakeBorder(cell, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
            cell = cv2.resize(cell, (50, 50))  # Resize to a standard size

            # Run OCR on the cell
            custom_config = r"--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(cell, config=custom_config).strip()

            # Parse OCR result
            if text.isdigit():
                sudoku_grid[row, col] = int(text)

    return sudoku_grid


def is_valid(board, row, col, num):
    """Check if a number can be placed in a specific position."""
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    subgrid_row_start = (row // 3) * 3
    subgrid_col_start = (col // 3) * 3
    subgrid = board[
        subgrid_row_start : subgrid_row_start + 3,
        subgrid_col_start : subgrid_col_start + 3,
    ]
    return num not in subgrid


def solve_sudoku(board):
    """Solve the Sudoku board using backtracking."""
    for row in range(9):
        for col in range(9):
            if board[row, col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row, col] = num
                        if solve_sudoku(board):
                            return True
                        board[row, col] = 0
                return False
    return True


def display_grid_with_colors(grid, title="Sudoku Grid"):
    """Display the Sudoku grid with background colors for non-empty cells."""
    st.write(f"### {title}")
    grid_html = "<table style='border-collapse: collapse;'>"

    for row in grid:
        grid_html += "<tr>"
        for cell in row:
            # Add background color for non-empty cells
            color = "#ADD8E6" if cell != 0 else "#FFFFFF"
            grid_html += f"<td style='border: 1px solid black; width: 30px; height: 30px; text-align: center; background-color: {color};'>{cell if cell != 0 else ''}</td>"
        grid_html += "</tr>"

    grid_html += "</table>"
    st.markdown(grid_html, unsafe_allow_html=True)


def main():
    st.title("Sudoku Solver with OCR")
    st.write(
        "Upload a Sudoku puzzle image to extract the grid automatically, or fill it manually."
    )

    # File upload for Sudoku image
    uploaded_file = st.file_uploader(
        "Upload an image of a Sudoku puzzle:", type=["png", "jpg", "jpeg"]
    )

    grid = np.zeros((9, 9), dtype=int)

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load and preprocess the image
        image = cv2.imread(temp_file_path)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        preprocessed_image = preprocess_image(image)
        st.image(
            preprocessed_image, caption="Preprocessed Image", use_container_width=True
        )

        # Extract Sudoku grid cells and recognize digits
        sudoku_grid = extract_sudoku_grid_cells(preprocessed_image)
        display_grid_with_colors(sudoku_grid, title="Extracted Sudoku Grid")

        # Allow manual correction of extracted grid
        st.write("### You can also fill the grid manually:")
        for i in range(9):
            cols = st.columns(9)
            for j, col in enumerate(cols):
                grid[i, j] = col.number_input(
                    f"Cell ({i+1},{j+1})",
                    min_value=0,
                    max_value=9,
                    value=int(sudoku_grid[i, j]),
                    key=f"cell_{i}_{j}",
                )

    if st.button("Solve"):
        if solve_sudoku(grid):
            display_grid_with_colors(grid, title="Solved Sudoku Grid")
        else:
            st.error("No solution exists for the given Sudoku puzzle.")


if __name__ == "__main__":
    main()