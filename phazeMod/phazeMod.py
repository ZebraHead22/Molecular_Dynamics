import re
import os
import ui
import sys
import numpy as np
import scipy as sp
from scipy import fftpack
from PyQt5 import QtWidgets
from scipy.fft import rfft, rfftfreq

class MainApplication(QtWidgets.QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.coeff_dial.valueChanged.connect(self.your_update_function)
        self.your_update_function(self.coeff_dial.value())

    def your_update_function(self, value):
       print("The value of your_dial is: ", value)
       self.lcdNumber.display(value)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApplication()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
