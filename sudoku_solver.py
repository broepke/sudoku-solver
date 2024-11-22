import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import tensorflow as tf
from tensorflow import keras
import os

def order_points(pts):
    """Order points in the order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    # Sum and difference of the points
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left point has the smallest sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right point has the largest sum
    rect[1] = pts[np.argmin(diff)]  # Top-right point has the smallest difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-left point has the largest difference
    return rect

def preprocess_image(image):
    """Preprocess the image for better grid detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return processed

def extract_sudoku_grid(image):
    """Extract the Sudoku grid from the processed image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Sort contours by area and take the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudoku_contour = contours[0]

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(sudoku_contour, True)
    approx = cv2.approxPolyDP(sudoku_contour, epsilon, True)

    if len(approx) == 4:  # Ensure the contour has 4 corners
        points = np.array([point[0] for point in approx], dtype="float32")
        # Order the points consistently
        points = order_points(points)
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

def remove_grid_lines(image):
    """Remove grid lines from the Sudoku grid image using combined methods."""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Threshold to create a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Remove horizontal and vertical lines using morphological operations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # Detect horizontal lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # Detect vertical lines
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine detected lines
    grid_lines = cv2.add(horizontal_lines, vertical_lines)

    # Dilate the grid lines to cover the thickness
    kernel = np.ones((3, 3), np.uint8)
    grid_lines = cv2.dilate(grid_lines, kernel, iterations=2)

    # Inpaint the original image
    image_without_grid = cv2.inpaint(gray, grid_lines, 5, cv2.INPAINT_TELEA)

    return image_without_grid

def extract_numbers_from_grid(grid, mnist_model):
    """Use CNN model to extract numbers from the Sudoku grid."""
    grid = remove_grid_lines(grid)
    st.image(grid, caption="Grid without Lines", use_column_width=True)

    cell_size = grid.shape[0] // 9
    sudoku_grid = np.zeros((9, 9), dtype=int)

    for row in range(9):
        for col in range(9):
            # Extract the cell image
            x_start, y_start = col * cell_size, row * cell_size
            x_end, y_end = x_start + cell_size, y_start + cell_size
            cell = grid[y_start:y_end, x_start:x_end]

            # Preprocess the cell image
            cell = cv2.resize(cell, (28, 28))
            cell = cv2.GaussianBlur(cell, (3, 3), 0)
            _, cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cell = cv2.bitwise_not(cell)  # Invert colors
            cell = cell / 255.0  # Normalize pixel values
            cell = cell.reshape(1, 28, 28, 1)

            # Predict using the CNN model
            prediction = mnist_model.predict(cell)
            digit = np.argmax(prediction)

            # Set a confidence threshold to distinguish empty cells
            confidence = np.max(prediction)
            if confidence > 0.8 and digit != 0:
                sudoku_grid[row, col] = digit
            else:
                sudoku_grid[row, col] = 0

            # Display the cell image and prediction for debugging
            cell_display = cell.reshape(28, 28)
            st.image(cell_display, caption=f"Cell ({row+1},{col+1}) Prediction: '{digit}', Confidence: {confidence:.2f}", width=100)

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
                        board[row, col] = 0  # Backtrack
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

        # Load and preprocess the image
        image = cv2.imread(temp_file_path)
        preprocessed_image = preprocess_image(image)
        st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)

        # Extract Sudoku grid
        sudoku_grid_image = extract_sudoku_grid(preprocessed_image)

        if sudoku_grid_image is not None:
            st.image(sudoku_grid_image, caption="Detected Sudoku Grid", use_column_width=True)

            # Load the CNN model
            if os.path.exists('mnist_cnn.h5'):
                mnist_model = keras.models.load_model('mnist_cnn.h5')
            else:
                st.error("CNN model not found. Please ensure 'mnist_cnn.h5' is in the same directory.")
                return

            grid = extract_numbers_from_grid(sudoku_grid_image, mnist_model=mnist_model)
            st.write("### Extracted Sudoku Grid:")
            st.write(grid)
        else:
            st.error("Could not detect a Sudoku grid. Please try another image.")

    # Manual input for Sudoku grid
    st.write("### You can also fill the grid manually:")
    for i in range(9):
        cols = st.columns(9)
        for j, col in enumerate(cols):
            grid[i, j] = col.number_input(f"Cell ({i+1},{j+1})", min_value=0, max_value=9, value=int(grid[i, j]), key=f"cell_{i}_{j}")

    if st.button("Solve"):
        original_grid = grid.copy()
        if solve_sudoku(grid):
            st.write("### Solved Sudoku:")
            st.write(grid)
        else:
            st.error("No solution exists for the given Sudoku puzzle.")

        st.write("### Original Puzzle:")
        st.write(original_grid)

if __name__ == "__main__":
    main()