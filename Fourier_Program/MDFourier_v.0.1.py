import re #import modules
import pandas
import Fourier
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from pandas import Series
from scipy import fftpack
from PyQt5 import QtWidgets
#-----------------------------------------------------------------------------------------------------------------------
class MainApplication(QtWidgets.QMainWindow, Fourier.Ui_MDFourier): #main application window
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.directory = False
        self.data = None
        self.times = []
        self.energies = []
        self.uploadbtn.clicked.connect(self.upload)
        # self.createMenuBar()
        self.showBtn.clicked.connect(self.goProcess)
        self.saveBtn.clicked.connect(self.saveData)
        self.extBtn.clicked.connect(self.exitMode)
        self.openBtn.clicked.connect(self.openFiles)
        self.openLogBtn.clicked.connect(self.brwLog)
        self.openDatBtn.clicked.connect(self.brwDat)
        self.dolboebBtn.clicked.connect(self.dolboeb)
# # ----------------------------------------------------------------------------------------------------------------------
#     def createMenuBar(self):
#         self.menuBar = QMenuBar(self)
#         self.setMenuBar(self.menuBar)
#         self.fileMenu = QMenu("&File", self)
#         self.menuBar.addMenu(self.fileMenu)
#         self.open_file = self.fileMenu.addMenu("&Open Data")
#         self.fileMenu.addAction("Save Data File", self.saveData)
#         self.fileMenu.addAction("End program", self.exitMode)
#         self.open_file.addAction("Automatic opening", self.uploadData)
#         self.open_file.addAction("Open dat file", self.brwDat)
#         self.open_file.addAction("Open log file", self.brwLog)
# ----------------------------------------------------------------------------------------------------------------------
    def openFiles(self):  # automatic upload files
        self.graphicsView_fourier.clear()
        self.graphicsView_energy.clear()
        self.times = []
        self.energies = []
        self.atomLabel.setText("----------------------------")
        self.srLabel.setText("----------------------------")
        self.durLabel.setText("----------------------------")
        self.mlLabel.setText("----------------------------")
        self.tsLabel.setText("----------------------------")
        self.datafile = None
        self.datLabel.setText("----------------------------")
        self.logLabel.setText("----------------------------")
        self.logfile = None

        self.termStatus.setText("Hello! Open folder to data process...")
        self.directory = QtWidgets.QFileDialog.getExistingDirectory(self)
        self.files = os.listdir(self.directory)
        for file in self.files:
            filename, file_extension = os.path.splitext(self.directory + "/" + file)
            if file_extension == ".dat":
                self.datafile =os.path.splitext(self.directory + "/" + file)[0]+".dat"
                self.termStatus.setText(self.datafile+" is uploaded...")
                if os.path.isfile(self.datafile):
                    if len(self.datafile) > 50:
                        self.datLabel.setText("..."+self.datafile[20:])
                    else:self.datLabel.setText(self.datafile)
                else:
                    self.datLabel.setText("File not found")
            if file_extension == ".log":
                self.logfile = os.path.splitext(self.directory + "/" + file)[0]+".log"
                self.termStatus.setText(self.logfile+" is uploaded...")
        if os.path.isfile(self.logfile):
            if len(self.datafile) > 50:
                self.logLabel.setText("..." + self.logfile[20:])
            else:
                self.logLabel.setText(self.logfile)
        else:
            self.logLabel.setText("File not found")
        os.chdir(self.directory)
        self.setWindowTitle("MDFourier in "+self.directory)

# ----------------------------------------------------------------------------------------------------------------------
    def brwLog(self):  # manual upload files
        self.logLabel.setText("----------------------------")
        self.times = []
        self.energies = []
        self.atomLabel.setText("----------------------------")
        self.srLabel.setText("----------------------------")
        self.durLabel.setText("----------------------------")
        self.mlLabel.setText("----------------------------")
        self.tsLabel.setText("----------------------------")
        self.logfile = None
        self.logfile = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if len(self.logfile) > 50:
            self.logLabel.setText("..." + self.logfile[20:])
        else:
            self.logLabel.setText(self.logfile)
        self.directory = os.path.dirname(self.logfile)
        self.termStatus.setText(self.directory)
        # ----------------------------------------------------------------------------------------------------------------------
    def brwDat(self):
        self.datafile = None
        self.datLabel.setText("----------------------------")
        self.datafile = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        if len(self.datafile) > 50:
            self.datLabel.setText("..." + self.datafile[20:])
        else:
            self.datLabel.setText(self.datafile)
        self.directoryNew = os.path.dirname(self.datfile)
        if self.directory == self.directoryNew:
            self.termStatus.setText("True")
#-----------------------------------------------------------------------------------------------------------------------
    def upload(self):
        self.graphicsView_energy.clear()
        self.times = []
        self.energies = []
        try:
            with open(self.logfile, 'r') as file: #Define ions
                line = file.readlines()  # [114]
                for i in line:
                    if re.findall(r'Info: \d+ ATOMS\n', i):
                        ions = re.search(r'\d+', i)
                        ions = int(ions.group(0))
                self.atomLabel.setText(str(ions))
        except AttributeError:
            pass
#-----------------------------------------------------------------------------------------------------------------------
        try:
            with open(self.logfile, 'r') as file: #Define duration time
                lastLine = file.readlines()[-4]
                duration = re.findall(r'\d+', lastLine)
                self.duration = int(duration[0])
                NS = round(float(self.duration/1000000), 3)
            self.durLabel.setText(str(NS) + " ns")
        except AttributeError:
            pass
#-----------------------------------------------------------------------------------------------------------------------
        with open(self.datafile, 'r') as f: #Define E & t to arrays
            my_lines = f.readlines()
        for i in my_lines:
            raw_time = re.findall(r'\d+ ', i)
            for a in raw_time:
                self.times.append(int(a) / 1000000)
            raw_energy = re.findall(r'(?<=\s).*\d+.\d+', i)
            for b in raw_energy:
                self.energies.append(float(b) / ions * 0.0434)
        self.mlLabel.setText(str(len(self.times))) # Number of exp
        self.cutTime = self.times[1]-self.times[0]
        self.tsLabel.setText(str(self.cutTime)+" ns")
        self.srLabel.setText(str(round(1/self.cutTime/(1000)))+" THz")
        self.termStatus.setText(str(len(self.times)) + " " + str(len(self.energies)))
        if len(self.times) == len(self.energies):
            self.termStatus.setText("OK")
        #plotting energy graph
        self.graphicsView_energy.setBackground('w')
        self.graphicsView_energy.plot(self.times, self.energies, pen='b', )
        self.termStatus.setText("Ready to Fourier...Change fuction")

# ----------------------------------------------------------------------------------------------------------------------
    def dolboeb(self):
        self.atoms = float(self.atoms.value())
        self.csvfile = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file')[0]
        self.termStatus.setText("WAIT")
        self.df = pd.read_csv(self.csvfile)
        self.newTimes = self.df['TS']
        self.newEnergies = self.df['ELECT']
        self.newTimes = np.array(Series.tolist(self.newTimes))
        self.newEnergies = np.array(Series.tolist(self.newEnergies))
        self.newTimes = self.newTimes / 1000000
        self.newEnergies = self.newEnergies / self.atoms * 0.0434
        cutTime = self.newTimes[1] - self.newTimes[0]
        self.graphicsView_energy.setBackground('w')
        self.graphicsView_energy.setLabel('bottom', 'time', units='ns')
        self.graphicsView_energy.setLabel('left', 'Energy', units='eV')
        self.graphicsView_energy.plot(self.newTimes, self.newEnergies, pen='b')
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        self.newy_gauss = np.array(
            gaussian(self.newTimes, float(self.newTimes[int(len(self.newTimes) / 2)]), 0.3))
        self.y_res = self.newEnergies * self.newy_gauss
        self.energies_fft = sp.fftpack.fft(np.array(self.y_res))
        self.energies_psd = np.abs(self.energies_fft)
        self.fftFreq = sp.fftpack.fftfreq(len(self.energies_fft), 1 / float(round(1 / cutTime * (10 ** 9))))
        self.i = self.fftFreq > 0
        self.fftFreqcm = 1 / ((2.9979 * (10 ** 10)) / self.fftFreq[self.i])
        self.graphicsView_fourier.setBackground('w')
        self.graphicsView_fourier.setLabel('bottom', 'Frequency (1/cm)', units =None)
        self.graphicsView_fourier.setLabel('left', 'Amplitude', units =None)
        self.graphicsView_fourier.plot(self.fftFreqcm, 10 * np.log10(self.energies_psd[self.i]), pen='r')
#-----------------------------------------------------------------------------------------------------------------------
    def goProcess(self): #Processing function
        self.graphicsView_fourier.clear()#Очищаем окно фурье картинки
        try:
            self.delValue = int(self.delValue.value())
            print("Deleted "+str(self.delValue))
        except AttributeError:
            self.delValue = 0
            print("Nothing to delete or self.value = 0")
        del self.times[:self.delValue]
        del self.energies[:self.delValue]
        self.xSamp = []
        self.ySamp = []
        self.sampleRate = round(1 / self.cutTime*(10**9))
        self.xSamp = np.array(self.times)#преобразуем х массив в np
        self.ySamp = np.array(self.energies)#тоже самое с У

        #Проверяем чекбокс
        if self.gaussBox.isChecked():
            def gaussian(x, mu, sig):
                return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))                                          #Мат функция гаусса
            self.sigmaValue = float(self.sigEdit.value())                                                               #Читаем сигму из спинбокса
            self.y_gauss = np.array(gaussian(self.xSamp, float(self.times[int(len(self.times) / 2)]), self.sigmaValue)) #Делаем массив с Гауссом (х-точки, центр, сигма)
            y_res = self.ySamp * self.y_gauss                                                                                #Считаем результат умножения Гаусса на наши энергии
            energies_fft = sp.fftpack.fft(np.array(y_res))                                                              #Берем БПФ от результирующей
            self.energies_psd = np.abs(energies_fft)                                                                    #А теперь модули
            self.fftFreq = sp.fftpack.fftfreq(len(energies_fft), 1 / float(self.sampleRate))                            #Делаем частоты
            self.i = self.fftFreq > 0                                                                                   #Выбираем все частоты больше нуля
            self.graphicsView_fourier.setBackground('w')
            if self.naturalBox.isChecked():
                self.graphicsView_fourier.plot(self.fftFreq[self.i], self.energies_psd[self.i], pen='r')
            if self.logBox.isChecked():
                self.graphicsView_fourier.plot(self.fftFreq[self.i], np.log10(self.energies_psd[self.i]), pen='b')
            if self.tenLogsBox.isChecked():
                self.graphicsView_fourier.plot(self.fftFreq[self.i], 10 * np.log10(self.energies_psd[self.i]), pen='g')    #Строим график спектра
            self.termStatus.setText("Ready to save data")
        #Строим график спектра
        else:
            energies_fft = sp.fftpack.fft(np.array(self.ySamp))#
            self.energies_psd = np.abs(energies_fft)#
            self.fftFreq = sp.fftpack.fftfreq(len(energies_fft), 1 / float(self.sampleRate)) #
            self.i = self.fftFreq > 0#
            self.graphicsView_fourier.setBackground('w')
            self.graphicsView_fourier.plot(self.fftFreq[self.i], 10 * np.log10(self.energies_psd[self.i]), pen='g')#
            self.termStatus.setText("Ready to save data")
# ----------------------------------------------------------------------------------------------------------------------
    def saveData(self): #save data
        df = pandas.DataFrame()
        df['f (Hz)'] = (self.fftFreq[self.i])/float(10**12)
        df['AmplitudePure'] = self.energies_psd[self.i]
        df['AmplitudeLog10'] = np.log10(self.energies_psd[self.i])
        df['Amplitude10Log10'] = 10 * np.log10(self.energies_psd[self.i])
        te = pandas.DataFrame()
        te['t (ns)'] = self.times
        te['E (eV)'] = self.energies
        ga = pandas.DataFrame()
        ga['Amplitude'] = self.xSamp
        ga['t (ns)'] = self.y_gauss
        with pd.ExcelWriter(str(self.directory)+"/"+'result.xlsx') as writer:
            df.to_excel(writer, sheet_name='fft', index=None, index_label=None)
            te.to_excel(writer, sheet_name='energy', index=None, index_label=None)
            ga.to_excel(writer, sheet_name='gauss', index=None, index_label=None)
        self.graphicsView_fourier.clear()
        self.graphicsView_energy.clear()
        self.times=[]
        self.energies=[]
        self.atomLabel.setText("----------------------------")
        self.srLabel.setText("----------------------------")
        self.durLabel.setText("----------------------------")
        self.mlLabel.setText("----------------------------")
        self.tsLabel.setText("----------------------------")
        self.datafile = None
        self.logfile = None
        self.datLabel.setText("----------------------------")
        self.logLabel.setText("----------------------------")
        self.termStatus.setText("Ready to upload new data")
        self.directory = None
# ----------------------------------------------------------------------------------------------------------------------
    def exitMode(self):
        exit()
# ----------------------------------------------------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = MainApplication()  # Создаём объект класса MainApplication
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение
if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()