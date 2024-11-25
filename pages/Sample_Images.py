import streamlit as st

# Streamlit page
st.title("Sudoku Solver Sample Images")

st.write("Use these sample images to test the Sudoku Solver application.")

# Display and provide a download link for the easy Sudoku image
st.header("Sample: Easy Sudoku")
st.image("sudoku_easy.png", caption="Easy Sudoku Puzzle", use_container_width=True)


# Display and provide a download link for the hard Sudoku image
st.header("Sample: Hard Sudoku")
st.image("sudoku_hard.png", caption="Hard Sudoku Puzzle", use_container_width=True)
