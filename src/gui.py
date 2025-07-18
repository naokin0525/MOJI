"""
GUI for Interactive Handwriting Generation.

This script provides a simple graphical user interface using PyQt5 to interact
with the handwriting generation model in real-time.
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QSlider, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QByteArray
import torch
import numpy as np

# This is a bit of a hack to make the script runnable from the root directory
# and find the 'src' package. A proper package installation would be better.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.vae_gan import VAE_GAN
from src.generate import sequence_to_svg_path
from src.config import DEVICE, LATENT_DIM, DEFAULT_LINE_HEIGHT, DEFAULT_CHAR_SPACING

class HandwritingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Handwriting Synthesizer")
        self.setGeometry(100, 100, 800, 600)

        # Model and state
        self.model = None
        self.current_svg_data = ""

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side - Controls
        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout, 1)

        # Right side - SVG Preview
        self.svg_preview = QSvgWidget()
        main_layout.addWidget(self.svg_preview, 2)

        # --- Controls ---
        # Model loading
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        controls_layout.addWidget(self.load_model_button)
        
        self.model_label = QLabel("No model loaded.")
        controls_layout.addWidget(self.model_label)
        
        controls_layout.addStretch()

        # Text input
        controls_layout.addWidget(QLabel("Text to Generate:"))
        self.text_input = QTextEdit("Hello World")
        self.text_input.setFixedHeight(100)
        controls_layout.addWidget(self.text_input)

        # Style controls
        controls_layout.addWidget(QLabel("Random Variation:"))
        self.variation_slider = QSlider(Qt.Horizontal)
        self.variation_slider.setRange(0, 100) # Representing 0.0 to 1.0
        self.variation_slider.setValue(5)
        controls_layout.addWidget(self.variation_slider)

        controls_layout.addWidget(QLabel("Stroke Width:"))
        self.stroke_slider = QSlider(Qt.Horizontal)
        self.stroke_slider.setRange(5, 50) # Representing 0.5 to 5.0
        self.stroke_slider.setValue(10)
        controls_layout.addWidget(self.stroke_slider)

        controls_layout.addStretch()
        
        # Action buttons
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate)
        self.generate_button.setEnabled(False) # Disabled until model is loaded
        controls_layout.addWidget(self.generate_button)

        self.save_button = QPushButton("Save SVG")
        self.save_button.clicked.connect(self.save_svg)
        controls_layout.addWidget(self.save_button)
        

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "PyTorch Models (*.pth)")
        if file_path:
            try:
                self.model = VAE_GAN()
                self.model.load_state_dict(torch.load(file_path, map_location=DEVICE))
                self.model.to(DEVICE)
                self.model.eval()
                self.model_label.setText(f"Loaded: {os.path.basename(file_path)}")
                self.generate_button.setEnabled(True)
                QMessageBox.information(self, "Success", "Model loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def generate(self):
        if not self.model:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return
        
        text = self.text_input.toPlainText()
        if not text:
            return
            
        variation = self.variation_slider.value() / 100.0
        stroke_width = self.stroke_slider.value() / 10.0

        full_svg_content = ''
        current_x_offset = 0

        with torch.no_grad():
            for char in text:
                if char.isspace():
                    current_x_offset += DEFAULT_CHAR_SPACING * 3
                    continue
                
                z = torch.randn(1, LATENT_DIM).to(DEVICE) + variation * torch.randn(1, LATENT_DIM).to(DEVICE)
                generated_sequence = self.model.generator(z, max_seq_len=200)
                sequence_np = generated_sequence.cpu().squeeze(0).numpy()
                svg_path = sequence_to_svg_path(sequence_np, stroke_width)
                full_svg_content += f'<g transform="translate({current_x_offset}, 40)">{svg_path}</g>\n'
                max_x = np.cumsum(sequence_np[:, 0]).max()
                current_x_offset += max_x + DEFAULT_CHAR_SPACING
        
        # Create SVG and render
        svg_header = f'<svg width="{current_x_offset}" height="{DEFAULT_LINE_HEIGHT}" xmlns="http://www.w3.org/2000/svg">\n'
        self.current_svg_data = svg_header + full_svg_content + '</svg>'
        self.svg_preview.load(QByteArray(self.current_svg_data.encode('utf-8')))

    def save_svg(self):
        if not self.current_svg_data:
            QMessageBox.warning(self, "Warning", "Nothing to save. Please generate handwriting first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save SVG", "", "SVG Files (*.svg)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.current_svg_data)
                QMessageBox.information(self, "Success", f"SVG saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandwritingGUI()
    window.show()
    sys.exit(app.exec_())