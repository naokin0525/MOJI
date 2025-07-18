"""
Defines the main GUI window for the dataset generation tool.
"""
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from .drawing_canvas import DrawingCanvas
from .svg_exporter import strokes_to_svg

class DatasetGeneratorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handwriting Dataset Generator")
        self.setMinimumSize(600, 500)

        # State
        self.save_directory = ""

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top Controls ---
        top_controls_layout = QHBoxLayout()
        self.dir_label = QLabel("Save Directory: Not Selected")
        self.dir_button = QPushButton("Select Directory")
        self.dir_button.clicked.connect(self.select_directory)
        top_controls_layout.addWidget(self.dir_label)
        top_controls_layout.addStretch()
        top_controls_layout.addWidget(self.dir_button)
        main_layout.addLayout(top_controls_layout)

        # --- Drawing Canvas ---
        self.canvas = DrawingCanvas()
        main_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)

        # --- Bottom Controls ---
        bottom_controls_layout = QHBoxLayout()
        bottom_controls_layout.addWidget(QLabel("Character to Draw:"))
        self.char_input = QLineEdit()
        self.char_input.setFixedWidth(50)
        self.char_input.setMaxLength(1) # Allow only one character
        
        self.save_button = QPushButton("Save & Clear")
        self.save_button.clicked.connect(self.save_drawing)
        
        self.clear_button = QPushButton("Clear Canvas")
        self.clear_button.clicked.connect(self.canvas.clear_canvas)

        bottom_controls_layout.addStretch()
        bottom_controls_layout.addWidget(QLabel("Character:"))
        bottom_controls_layout.addWidget(self.char_input)
        bottom_controls_layout.addWidget(self.clear_button)
        bottom_controls_layout.addWidget(self.save_button)
        bottom_controls_layout.addStretch()
        main_layout.addLayout(bottom_controls_layout)

    def select_directory(self):
        """Opens a dialog to choose a directory for saving SVG files."""
        directory = QFileDialog.getExistingDirectory(self, "Select Folder")
        if directory:
            self.save_directory = directory
            # Truncate label if path is too long
            display_path = self.save_directory
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]
            self.dir_label.setText(f"Save Directory: {display_path}")

    def save_drawing(self):
        """Validates input and saves the current drawing as an SVG file."""
        char = self.char_input.text()
        strokes = self.canvas.get_strokes()

        # --- Validation ---
        if not self.save_directory:
            QMessageBox.warning(self, "Error", "Please select a save directory first.")
            return
        if not char:
            QMessageBox.warning(self, "Error", "Please enter the character you drew.")
            return
        if not strokes:
            QMessageBox.warning(self, "Error", "The canvas is empty. Please draw a character.")
            return

        # --- SVG Export ---
        try:
            filename = f"{char}.svg"
            filepath = os.path.join(self.save_directory, filename)

            svg_content = strokes_to_svg(strokes, self.canvas.width(), self.canvas.height())

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(svg_content)
            
            QMessageBox.information(self, "Success", f"Saved drawing to:\n{filepath}")
            
            # Reset for next character
            self.canvas.clear_canvas()
            self.char_input.clear()
            self.char_input.setFocus()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save the file: {e}")