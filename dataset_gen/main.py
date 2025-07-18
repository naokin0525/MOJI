"""
Main entry point for the Dataset Generation GUI application.
"""
import sys
from PyQt5.QtWidgets import QApplication
from .gui import DatasetGeneratorGUI

def main():
    """
    Initializes and runs the PyQt5 application.
    """
    app = QApplication(sys.argv)
    window = DatasetGeneratorGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()