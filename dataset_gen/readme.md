# Handwriting Dataset Generation Tool

This is a simple graphical user interface (GUI) tool for creating handwritten character datasets in SVG format. It allows you to draw characters on a canvas and save them with the correct filename required for the main AI training program.

## Features

- **Drawing Canvas**: A simple white canvas to draw characters with your mouse or a graphics tablet.
- **Character Input**: A text field to specify the character you are drawing.
- **Automatic SVG Export**: Saves the drawing as an SVG file with the character as the filename (e.g., `a.svg`, `字.svg`).
- **Directory Selection**: A dialog to choose where to save the dataset files.

## Requirements

This tool requires the `PyQt5` library. If you have already installed the requirements for the main project, you are all set. Otherwise, install it via pip:

```sh
pip install PyQt5
```

## How to Use

1.  **Run the application:**
    Navigate to the project's root directory (`handwriting-synthesis/`) and run the `main.py` script from the `dataset_gen` directory:

    ```sh
    python dataset_gen/main.py
    ```

2.  **Select a Save Directory:**

    - Click the "Select Directory" button.
    - Choose an existing folder or create a new one where you want to store your SVG files (e.g., `data/raw/my_custom_dataset`).

3.  **Enter a Character:**

    - In the "Character to Draw" text box, type the single character you intend to draw (e.g., `b` or `猫`).

4.  **Draw the Character:**

    - Use your mouse to draw the character on the white canvas area. Each continuous stroke (from mouse-down to mouse-up) is recorded as a separate path.

5.  **Save the Drawing:**

    - Click the "Save & Clear" button.
    - The tool will save your drawing as an SVG file in the selected directory. For example, if you entered `猫`, the file will be named `猫.svg`.
    - The canvas will automatically be cleared, ready for you to draw the next character.

6.  **Repeat:**
    - Repeat steps 3-5 for every character you want to add to your dataset.
