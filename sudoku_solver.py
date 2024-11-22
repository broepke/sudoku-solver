import streamlit as st
import numpy as np
import easyocr
import cv2
from PIL import Image
import tempfile


def preprocess_image(image):
    """Preprocess the image for better OCR performance."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed


def extract_sudoku_grid(image):
    """Extract the Sudoku grid from the processed image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area and take the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudoku_contour = contours[0]

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(sudoku_contour, True)
    approx = cv2.approxPolyDP(sudoku_contour, epsilon, True)

    if len(approx) == 4:  # Ensure the contour has 4 corners
        points = np.float32([point[0] for point in approx])
        # Sort points in clockwise order
        points = points[np.argsort(points[:, 0])]
        points[:2] = points[np.argsort(points[:2][:, 1])]
        points[2:] = points[2 + np.argsort(points[2:][:, 1])]

        # Define the target square size
        side = max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])
        )
        target = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
        transform_matrix = cv2.getPerspectiveTransform(points, target)
        grid = cv2.warpPerspective(image, transform_matrix, (int(side), int(side)))
        return grid
    else:
        return None


def extract_numbers_from_grid(grid, reader):
    """Use OCR to extract numbers from the Sudoku grid."""
    cell_size = grid.shape[0] // 9
    sudoku_grid = np.zeros((9, 9), dtype=int)

    for row in range(9):
        for col in range(9):
            # Extract each cell
            x_start, y_start = col * cell_size, row * cell_size
            x_end, y_end = x_start + cell_size, y_start + cell_size
            cell = grid[y_start:y_end, x_start:x_end]

            # Run OCR on the cell
            ocr_result = reader.readtext(cell, detail=0)
            if ocr_result:
                try:
                    sudoku_grid[row, col] = int(ocr_result[0])  # Get the first detected number
                except ValueError:
                    pass  # Skip non-integer OCR results

    return sudoku_grid


def is_valid(board, row, col, num):
    """Check if a number can be placed in a specific position."""
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    subgrid_row_start = (row // 3) * 3
    subgrid_col_start = (col // 3) * 3
    subgrid = board[subgrid_row_start:subgrid_row_start + 3, subgrid_col_start:subgrid_col_start + 3]
    if num in subgrid:
        return False
    return True


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


def main():
    st.title("Sudoku Solver with Image Upload")
    st.write("Upload a Sudoku puzzle image to populate the grid automatically, or fill it manually.")

    # File upload for Sudoku image
    uploaded_file = st.file_uploader("Upload an image of a Sudoku puzzle:", type=["png", "jpg", "jpeg"])
    
    grid = np.zeros((9, 9), dtype=int)

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        image = cv2.imread(temp_file_path)
        preprocessed_image = preprocess_image(image)
        sudoku_grid_image = extract_sudoku_grid(preprocessed_image)

        if sudoku_grid_image is not None:
            reader = easyocr.Reader(['en'])
            grid = extract_numbers_from_grid(sudoku_grid_image, reader)
            st.write("Extracted Sudoku Grid:")
            st.write(grid)
        else:
            st.write("Could not detect a Sudoku grid. Please try another image.")

    # Manual input for Sudoku grid
    st.write("You can also fill the grid manually:")
    for i in range(9):
        cols = st.columns(9)
        for j, col in enumerate(cols):
            grid[i, j] = col.number_input(f"Cell ({i+1},{j+1})", min_value=0, max_value=9, value=grid[i, j], key=f"cell_{i}_{j}")

    if st.button("Solve"):
        original_grid = grid.copy()
        if solve_sudoku(grid):
            st.write("### Solved Sudoku")
            st.write(grid)
        else:
            st.write("No solution exists for the given Sudoku puzzle.")
        
        st.write("### Original Puzzle")
        st.write(original_grid)


if __name__ == "__main__":
    main()