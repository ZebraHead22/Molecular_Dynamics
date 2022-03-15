# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Fourier.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MDFourier(object):
    def setupUi(self, MDFourier):
        MDFourier.setObjectName("MDFourier")
        MDFourier.resize(1259, 819)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MDFourier.sizePolicy().hasHeightForWidth())
        MDFourier.setSizePolicy(sizePolicy)
        MDFourier.setMinimumSize(QtCore.QSize(100, 100))
        MDFourier.setMaximumSize(QtCore.QSize(5000, 4000))
        MDFourier.setStyleSheet("background-color: rgb(231, 236, 255)")
        self.centralwidget = QtWidgets.QWidget(MDFourier)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.TitleWidget = QtWidgets.QWidget(self.centralwidget)
        self.TitleWidget.setMinimumSize(QtCore.QSize(0, 40))
        self.TitleWidget.setMaximumSize(QtCore.QSize(16777215, 40))
        self.TitleWidget.setObjectName("TitleWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.TitleWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frameEnergyGraph = QtWidgets.QFrame(self.TitleWidget)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(14)
        self.frameEnergyGraph.setFont(font)
        self.frameEnergyGraph.setStyleSheet("border : none;")
        self.frameEnergyGraph.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameEnergyGraph.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameEnergyGraph.setObjectName("frameEnergyGraph")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frameEnergyGraph)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.labelEhergyGraph = QtWidgets.QLabel(self.frameEnergyGraph)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.labelEhergyGraph.setFont(font)
        self.labelEhergyGraph.setStyleSheet("color: rgb(81, 0, 255)")
        self.labelEhergyGraph.setAlignment(QtCore.Qt.AlignCenter)
        self.labelEhergyGraph.setObjectName("labelEhergyGraph")
        self.horizontalLayout_2.addWidget(self.labelEhergyGraph)
        self.horizontalLayout.addWidget(self.frameEnergyGraph)
        self.frameFourierGraph = QtWidgets.QFrame(self.TitleWidget)
        self.frameFourierGraph.setStyleSheet("color: rgb(81, 0, 255);\n"
"border : none;")
        self.frameFourierGraph.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameFourierGraph.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameFourierGraph.setObjectName("frameFourierGraph")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frameFourierGraph)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.FourierGraphLabelTitle = QtWidgets.QLabel(self.frameFourierGraph)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(13)
        self.FourierGraphLabelTitle.setFont(font)
        self.FourierGraphLabelTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.FourierGraphLabelTitle.setObjectName("FourierGraphLabelTitle")
        self.horizontalLayout_3.addWidget(self.FourierGraphLabelTitle)
        self.horizontalLayout.addWidget(self.frameFourierGraph)
        self.verticalLayout.addWidget(self.TitleWidget)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setMinimumSize(QtCore.QSize(0, 410))
        self.widget.setMaximumSize(QtCore.QSize(16777215, 410))
        self.widget.setObjectName("widget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.graphicsView_energy = PlotWidget(self.widget)
        self.graphicsView_energy.setMaximumSize(QtCore.QSize(595, 389))
        self.graphicsView_energy.setObjectName("graphicsView_energy")
        self.horizontalLayout_4.addWidget(self.graphicsView_energy)
        self.graphicsView_fourier = PlotWidget(self.widget)
        self.graphicsView_fourier.setMaximumSize(QtCore.QSize(595, 389))
        self.graphicsView_fourier.setObjectName("graphicsView_fourier")
        self.horizontalLayout_4.addWidget(self.graphicsView_fourier)
        self.verticalLayout.addWidget(self.widget)
        self.manageWidget = QtWidgets.QWidget(self.centralwidget)
        self.manageWidget.setMinimumSize(QtCore.QSize(0, 300))
        self.manageWidget.setMaximumSize(QtCore.QSize(16777215, 300))
        self.manageWidget.setStyleSheet("border:none;")
        self.manageWidget.setObjectName("manageWidget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.manageWidget)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.infoFrame = QtWidgets.QFrame(self.manageWidget)
        self.infoFrame.setMaximumSize(QtCore.QSize(595, 16777215))
        self.infoFrame.setStyleSheet("border : none;")
        self.infoFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.infoFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.infoFrame.setObjectName("infoFrame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.infoFrame)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.files = QtWidgets.QWidget(self.infoFrame)
        self.files.setMaximumSize(QtCore.QSize(16777215, 80))
        self.files.setObjectName("files")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.files)
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_14.setSpacing(0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.widget_14 = QtWidgets.QWidget(self.files)
        self.widget_14.setObjectName("widget_14")
        self.gridLayout = QtWidgets.QGridLayout(self.widget_14)
        self.gridLayout.setObjectName("gridLayout")
        self.logLabelTitle = QtWidgets.QLabel(self.widget_14)
        self.logLabelTitle.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.logLabelTitle.setFont(font)
        self.logLabelTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.logLabelTitle.setObjectName("logLabelTitle")
        self.gridLayout.addWidget(self.logLabelTitle, 0, 0, 1, 1)
        self.logLabel = QtWidgets.QLabel(self.widget_14)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.logLabel.setFont(font)
        self.logLabel.setStyleSheet("color: rgb(81, 0, 255)")
        self.logLabel.setObjectName("logLabel")
        self.gridLayout.addWidget(self.logLabel, 0, 1, 1, 1)
        self.datLabelTitle = QtWidgets.QLabel(self.widget_14)
        self.datLabelTitle.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.datLabelTitle.setFont(font)
        self.datLabelTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.datLabelTitle.setObjectName("datLabelTitle")
        self.gridLayout.addWidget(self.datLabelTitle, 1, 0, 1, 1)
        self.datLabel = QtWidgets.QLabel(self.widget_14)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.datLabel.setFont(font)
        self.datLabel.setStyleSheet("color: rgb(81, 0, 255)")
        self.datLabel.setObjectName("datLabel")
        self.gridLayout.addWidget(self.datLabel, 1, 1, 1, 1)
        self.horizontalLayout_14.addWidget(self.widget_14)
        self.widget_15 = QtWidgets.QWidget(self.files)
        self.widget_15.setObjectName("widget_15")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget_15)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.datDcdLabelTitle = QtWidgets.QLabel(self.widget_15)
        self.datDcdLabelTitle.setMaximumSize(QtCore.QSize(60, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.datDcdLabelTitle.setFont(font)
        self.datDcdLabelTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.datDcdLabelTitle.setObjectName("datDcdLabelTitle")
        self.gridLayout_4.addWidget(self.datDcdLabelTitle, 0, 0, 1, 1)
        self.datLabel_2 = QtWidgets.QLabel(self.widget_15)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.datLabel_2.setFont(font)
        self.datLabel_2.setStyleSheet("color: rgb(81, 0, 255)")
        self.datLabel_2.setObjectName("datLabel_2")
        self.gridLayout_4.addWidget(self.datLabel_2, 1, 1, 1, 1)
        self.csvLabelTitle = QtWidgets.QLabel(self.widget_15)
        self.csvLabelTitle.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.csvLabelTitle.setFont(font)
        self.csvLabelTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.csvLabelTitle.setObjectName("csvLabelTitle")
        self.gridLayout_4.addWidget(self.csvLabelTitle, 1, 0, 1, 1)
        self.logLabel_2 = QtWidgets.QLabel(self.widget_15)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.logLabel_2.setFont(font)
        self.logLabel_2.setStyleSheet("color: rgb(81, 0, 255)")
        self.logLabel_2.setObjectName("logLabel_2")
        self.gridLayout_4.addWidget(self.logLabel_2, 0, 1, 1, 1)
        self.horizontalLayout_14.addWidget(self.widget_15)
        self.verticalLayout_3.addWidget(self.files)
        self.widget_10 = QtWidgets.QWidget(self.infoFrame)
        self.widget_10.setObjectName("widget_10")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.widget_10)
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.widget_12 = QtWidgets.QWidget(self.widget_10)
        self.widget_12.setObjectName("widget_12")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget_12)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.atomLabelTitle = QtWidgets.QLabel(self.widget_12)
        self.atomLabelTitle.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.atomLabelTitle.setFont(font)
        self.atomLabelTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.atomLabelTitle.setObjectName("atomLabelTitle")
        self.gridLayout_3.addWidget(self.atomLabelTitle, 0, 0, 1, 1)
        self.atomLabel = QtWidgets.QLabel(self.widget_12)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.atomLabel.setFont(font)
        self.atomLabel.setStyleSheet("color: rgb(81, 0, 255)")
        self.atomLabel.setObjectName("atomLabel")
        self.gridLayout_3.addWidget(self.atomLabel, 0, 1, 1, 1)
        self.durationLabelTitle = QtWidgets.QLabel(self.widget_12)
        self.durationLabelTitle.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.durationLabelTitle.setFont(font)
        self.durationLabelTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.durationLabelTitle.setObjectName("durationLabelTitle")
        self.gridLayout_3.addWidget(self.durationLabelTitle, 1, 0, 1, 1)
        self.durLabel = QtWidgets.QLabel(self.widget_12)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.durLabel.setFont(font)
        self.durLabel.setStyleSheet("color: rgb(81, 0, 255)")
        self.durLabel.setObjectName("durLabel")
        self.gridLayout_3.addWidget(self.durLabel, 1, 1, 1, 1)
        self.tsLabelTitle = QtWidgets.QLabel(self.widget_12)
        self.tsLabelTitle.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.tsLabelTitle.setFont(font)
        self.tsLabelTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.tsLabelTitle.setObjectName("tsLabelTitle")
        self.gridLayout_3.addWidget(self.tsLabelTitle, 2, 0, 1, 1)
        self.tsLabel = QtWidgets.QLabel(self.widget_12)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.tsLabel.setFont(font)
        self.tsLabel.setStyleSheet("color: rgb(81, 0, 255)")
        self.tsLabel.setObjectName("tsLabel")
        self.gridLayout_3.addWidget(self.tsLabel, 2, 1, 1, 1)
        self.srLabelTitle = QtWidgets.QLabel(self.widget_12)
        self.srLabelTitle.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.srLabelTitle.setFont(font)
        self.srLabelTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.srLabelTitle.setObjectName("srLabelTitle")
        self.gridLayout_3.addWidget(self.srLabelTitle, 3, 0, 1, 1)
        self.srLabel = QtWidgets.QLabel(self.widget_12)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.srLabel.setFont(font)
        self.srLabel.setStyleSheet("color: rgb(81, 0, 255)")
        self.srLabel.setObjectName("srLabel")
        self.gridLayout_3.addWidget(self.srLabel, 3, 1, 1, 1)
        self.mlLabelTitle = QtWidgets.QLabel(self.widget_12)
        self.mlLabelTitle.setMaximumSize(QtCore.QSize(100, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.mlLabelTitle.setFont(font)
        self.mlLabelTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.mlLabelTitle.setObjectName("mlLabelTitle")
        self.gridLayout_3.addWidget(self.mlLabelTitle, 4, 0, 1, 1)
        self.mlLabel = QtWidgets.QLabel(self.widget_12)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.mlLabel.setFont(font)
        self.mlLabel.setStyleSheet("color: rgb(81, 0, 255)")
        self.mlLabel.setObjectName("mlLabel")
        self.gridLayout_3.addWidget(self.mlLabel, 4, 1, 1, 1)
        self.horizontalLayout_13.addWidget(self.widget_12)
        self.widget_13 = QtWidgets.QWidget(self.widget_10)
        self.widget_13.setObjectName("widget_13")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.widget_13)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.widget_17 = QtWidgets.QWidget(self.widget_13)
        self.widget_17.setObjectName("widget_17")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_17)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_2 = QtWidgets.QLabel(self.widget_17)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(81, 0, 255)")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_7.addWidget(self.label_2)
        self.widget_18 = QtWidgets.QWidget(self.widget_17)
        self.widget_18.setObjectName("widget_18")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.widget_18)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.atomNumValueTitle = QtWidgets.QLabel(self.widget_18)
        self.atomNumValueTitle.setMaximumSize(QtCore.QSize(40, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.atomNumValueTitle.setFont(font)
        self.atomNumValueTitle.setStyleSheet("color: rgb(81, 0, 255)")
        self.atomNumValueTitle.setObjectName("atomNumValueTitle")
        self.horizontalLayout_15.addWidget(self.atomNumValueTitle)
        self.atomNumValue = QtWidgets.QDoubleSpinBox(self.widget_18)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.atomNumValue.setFont(font)
        self.atomNumValue.setStyleSheet("color: rgb(81, 0, 255)")
        self.atomNumValue.setDecimals(0)
        self.atomNumValue.setMaximum(100000.0)
        self.atomNumValue.setSingleStep(1.0)
        self.atomNumValue.setObjectName("atomNumValue")
        self.horizontalLayout_15.addWidget(self.atomNumValue)
        self.verticalLayout_7.addWidget(self.widget_18)
        self.verticalLayout_6.addWidget(self.widget_17)
        self.widget_16 = QtWidgets.QWidget(self.widget_13)
        self.widget_16.setObjectName("widget_16")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget_16)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_3 = QtWidgets.QLabel(self.widget_16)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: rgb(81, 0, 255)")
        self.label_3.setObjectName("label_3")
        self.verticalLayout_8.addWidget(self.label_3)
        self.widget_19 = QtWidgets.QWidget(self.widget_16)
        self.widget_19.setObjectName("widget_19")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.widget_19)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.atomNumValue_2 = QtWidgets.QLabel(self.widget_19)
        self.atomNumValue_2.setMinimumSize(QtCore.QSize(40, 0))
        self.atomNumValue_2.setMaximumSize(QtCore.QSize(40, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.atomNumValue_2.setFont(font)
        self.atomNumValue_2.setStyleSheet("color: rgb(81, 0, 255)")
        self.atomNumValue_2.setObjectName("atomNumValue_2")
        self.horizontalLayout_16.addWidget(self.atomNumValue_2)
        self.srNumValue = QtWidgets.QDoubleSpinBox(self.widget_19)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.srNumValue.setFont(font)
        self.srNumValue.setStyleSheet("color: rgb(81, 0, 255)")
        self.srNumValue.setDecimals(0)
        self.srNumValue.setMaximum(100000.0)
        self.srNumValue.setSingleStep(1.0)
        self.srNumValue.setObjectName("srNumValue")
        self.horizontalLayout_16.addWidget(self.srNumValue)
        self.verticalLayout_8.addWidget(self.widget_19)
        self.verticalLayout_6.addWidget(self.widget_16)
        self.horizontalLayout_13.addWidget(self.widget_13)
        self.verticalLayout_3.addWidget(self.widget_10)
        self.horizontalLayout_5.addWidget(self.infoFrame)
        self.manageFrame = QtWidgets.QFrame(self.manageWidget)
        self.manageFrame.setMaximumSize(QtCore.QSize(595, 16777215))
        self.manageFrame.setStyleSheet("color: rgb(81, 0, 255);\n"
"")
        self.manageFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.manageFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.manageFrame.setObjectName("manageFrame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.manageFrame)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_2 = QtWidgets.QWidget(self.manageFrame)
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 40))
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_6.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.gaussBox = QtWidgets.QCheckBox(self.widget_2)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.gaussBox.setFont(font)
        self.gaussBox.setStyleSheet("color: rgb(81, 0, 255)")
        self.gaussBox.setObjectName("gaussBox")
        self.horizontalLayout_6.addWidget(self.gaussBox)
        self.gaussBox_2 = QtWidgets.QCheckBox(self.widget_2)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.gaussBox_2.setFont(font)
        self.gaussBox_2.setStyleSheet("color: rgb(81, 0, 255)")
        self.gaussBox_2.setObjectName("gaussBox_2")
        self.horizontalLayout_6.addWidget(self.gaussBox_2)
        self.widget_23 = QtWidgets.QWidget(self.widget_2)
        self.widget_23.setObjectName("widget_23")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.widget_23)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.sigmaLabel_3 = QtWidgets.QLabel(self.widget_23)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.sigmaLabel_3.setFont(font)
        self.sigmaLabel_3.setStyleSheet("color: rgb(81, 0, 255)")
        self.sigmaLabel_3.setObjectName("sigmaLabel_3")
        self.horizontalLayout_17.addWidget(self.sigmaLabel_3)
        self.sigEdit_3 = QtWidgets.QDoubleSpinBox(self.widget_23)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.sigEdit_3.setFont(font)
        self.sigEdit_3.setStyleSheet("color: rgb(81, 0, 255)")
        self.sigEdit_3.setSingleStep(0.01)
        self.sigEdit_3.setObjectName("sigEdit_3")
        self.horizontalLayout_17.addWidget(self.sigEdit_3)
        self.horizontalLayout_6.addWidget(self.widget_23)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.widget_3 = QtWidgets.QWidget(self.manageFrame)
        self.widget_3.setMinimumSize(QtCore.QSize(0, 40))
        self.widget_3.setMaximumSize(QtCore.QSize(16777215, 40))
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.naturalBox = QtWidgets.QCheckBox(self.widget_3)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.naturalBox.setFont(font)
        self.naturalBox.setObjectName("naturalBox")
        self.horizontalLayout_7.addWidget(self.naturalBox)
        self.logBox = QtWidgets.QCheckBox(self.widget_3)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.logBox.setFont(font)
        self.logBox.setObjectName("logBox")
        self.horizontalLayout_7.addWidget(self.logBox)
        self.tenLogsBox = QtWidgets.QCheckBox(self.widget_3)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.tenLogsBox.setFont(font)
        self.tenLogsBox.setObjectName("tenLogsBox")
        self.horizontalLayout_7.addWidget(self.tenLogsBox)
        self.verticalLayout_2.addWidget(self.widget_3)
        self.widget_4 = QtWidgets.QWidget(self.manageFrame)
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.widget_5 = QtWidgets.QWidget(self.widget_4)
        self.widget_5.setObjectName("widget_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.openCSVBtn = QtWidgets.QPushButton(self.widget_5)
        self.openCSVBtn.setMinimumSize(QtCore.QSize(90, 30))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.openCSVBtn.setFont(font)
        self.openCSVBtn.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"    background-color:rgb(200, 192, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    \n"
"    background-color: rgb(198, 216, 255);\n"
"}")
        self.openCSVBtn.setObjectName("openCSVBtn")
        self.verticalLayout_4.addWidget(self.openCSVBtn)
        self.opendatlogBtn = QtWidgets.QPushButton(self.widget_5)
        self.opendatlogBtn.setMinimumSize(QtCore.QSize(90, 30))
        self.opendatlogBtn.setMaximumSize(QtCore.QSize(40404, 16777215))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.opendatlogBtn.setFont(font)
        self.opendatlogBtn.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"    background-color:rgb(200, 192, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    \n"
"    background-color: rgb(198, 216, 255);\n"
"}")
        self.opendatlogBtn.setObjectName("opendatlogBtn")
        self.verticalLayout_4.addWidget(self.opendatlogBtn)
        self.openDCDBtn_2 = QtWidgets.QPushButton(self.widget_5)
        self.openDCDBtn_2.setMinimumSize(QtCore.QSize(90, 30))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.openDCDBtn_2.setFont(font)
        self.openDCDBtn_2.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"    background-color:rgb(200, 192, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    \n"
"    background-color: rgb(198, 216, 255);\n"
"}")
        self.openDCDBtn_2.setObjectName("openDCDBtn_2")
        self.verticalLayout_4.addWidget(self.openDCDBtn_2)
        self.horizontalLayout_8.addWidget(self.widget_5)
        self.widget_7 = QtWidgets.QWidget(self.widget_4)
        self.widget_7.setObjectName("widget_7")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_7)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.uploadbtn = QtWidgets.QPushButton(self.widget_7)
        self.uploadbtn.setMinimumSize(QtCore.QSize(90, 30))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.uploadbtn.setFont(font)
        self.uploadbtn.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"    background-color:rgb(200, 192, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    \n"
"    background-color: rgb(198, 216, 255);\n"
"}")
        self.uploadbtn.setObjectName("uploadbtn")
        self.verticalLayout_5.addWidget(self.uploadbtn)
        self.showBtn = QtWidgets.QPushButton(self.widget_7)
        self.showBtn.setMinimumSize(QtCore.QSize(90, 30))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.showBtn.setFont(font)
        self.showBtn.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"    background-color:rgb(200, 192, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    \n"
"    background-color: rgb(198, 216, 255);\n"
"}")
        self.showBtn.setObjectName("showBtn")
        self.verticalLayout_5.addWidget(self.showBtn)
        self.horizontalLayout_8.addWidget(self.widget_7)
        self.widget_6 = QtWidgets.QWidget(self.widget_4)
        self.widget_6.setObjectName("widget_6")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.widget_6)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.saveBtn = QtWidgets.QPushButton(self.widget_6)
        self.saveBtn.setMinimumSize(QtCore.QSize(90, 30))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.saveBtn.setFont(font)
        self.saveBtn.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"    background-color:rgb(200, 192, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    \n"
"    background-color: rgb(198, 216, 255);\n"
"}")
        self.saveBtn.setObjectName("saveBtn")
        self.verticalLayout_9.addWidget(self.saveBtn)
        self.vmdBtn = QtWidgets.QPushButton(self.widget_6)
        self.vmdBtn.setMinimumSize(QtCore.QSize(90, 30))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.vmdBtn.setFont(font)
        self.vmdBtn.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"    background-color:rgb(200, 192, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    \n"
"    background-color: rgb(198, 216, 255);\n"
"}")
        self.vmdBtn.setObjectName("vmdBtn")
        self.verticalLayout_9.addWidget(self.vmdBtn)
        self.extBtn = QtWidgets.QPushButton(self.widget_6)
        self.extBtn.setMinimumSize(QtCore.QSize(90, 30))
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(10)
        self.extBtn.setFont(font)
        self.extBtn.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"    background-color:rgb(200, 192, 255);\n"
"}\n"
"QPushButton:hover {\n"
"    \n"
"    background-color: rgb(198, 216, 255);\n"
"}")
        self.extBtn.setObjectName("extBtn")
        self.verticalLayout_9.addWidget(self.extBtn)
        self.horizontalLayout_8.addWidget(self.widget_6)
        self.verticalLayout_2.addWidget(self.widget_4)
        self.horizontalLayout_5.addWidget(self.manageFrame)
        self.verticalLayout.addWidget(self.manageWidget)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        MDFourier.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MDFourier)
        self.statusbar.setObjectName("statusbar")
        MDFourier.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MDFourier)
        self.actionOpen.setObjectName("actionOpen")
        self.actionExport_CSV = QtWidgets.QAction(MDFourier)
        self.actionExport_CSV.setObjectName("actionExport_CSV")
        self.actionExport_All = QtWidgets.QAction(MDFourier)
        self.actionExport_All.setObjectName("actionExport_All")

        self.retranslateUi(MDFourier)
        QtCore.QMetaObject.connectSlotsByName(MDFourier)

    def retranslateUi(self, MDFourier):
        _translate = QtCore.QCoreApplication.translate
        MDFourier.setWindowTitle(_translate("MDFourier", "MDFourier"))
        self.labelEhergyGraph.setText(_translate("MDFourier", "Energy/DipMoments Graph"))
        self.FourierGraphLabelTitle.setText(_translate("MDFourier", "Fourier Graph"))
        self.logLabelTitle.setText(_translate("MDFourier", "log file:"))
        self.logLabel.setText(_translate("MDFourier", "----------------------------"))
        self.datLabelTitle.setText(_translate("MDFourier", "data file:"))
        self.datLabel.setText(_translate("MDFourier", "----------------------------"))
        self.datDcdLabelTitle.setText(_translate("MDFourier", "dat file:"))
        self.datLabel_2.setText(_translate("MDFourier", "----------------------------"))
        self.csvLabelTitle.setText(_translate("MDFourier", "csv file:"))
        self.logLabel_2.setText(_translate("MDFourier", "----------------------------"))
        self.atomLabelTitle.setText(_translate("MDFourier", "atoms:"))
        self.atomLabel.setText(_translate("MDFourier", "----------------------------"))
        self.durationLabelTitle.setText(_translate("MDFourier", "duration:"))
        self.durLabel.setText(_translate("MDFourier", "----------------------------"))
        self.tsLabelTitle.setText(_translate("MDFourier", "time step:"))
        self.tsLabel.setText(_translate("MDFourier", "----------------------------"))
        self.srLabelTitle.setText(_translate("MDFourier", "sample rate:"))
        self.srLabel.setText(_translate("MDFourier", "----------------------------"))
        self.mlLabelTitle.setText(_translate("MDFourier", "massive lenght:"))
        self.mlLabel.setText(_translate("MDFourier", "----------------------------"))
        self.label_2.setText(_translate("MDFourier", "Fast csv energy processing, set atom num:"))
        self.atomNumValueTitle.setText(_translate("MDFourier", "Atoms"))
        self.label_3.setText(_translate("MDFourier", "Dipole moments processing, set sample rate:"))
        self.atomNumValue_2.setText(_translate("MDFourier", "SR(fs)"))
        self.gaussBox.setText(_translate("MDFourier", "Hamming"))
        self.gaussBox_2.setText(_translate("MDFourier", "Sin"))
        self.sigmaLabel_3.setText(_translate("MDFourier", "Period"))
        self.naturalBox.setText(_translate("MDFourier", "Pure Energy"))
        self.logBox.setText(_translate("MDFourier", "Log 10"))
        self.tenLogsBox.setText(_translate("MDFourier", "10*Log10"))
        self.openCSVBtn.setText(_translate("MDFourier", "Open CSV"))
        self.opendatlogBtn.setText(_translate("MDFourier", "Open dat/log"))
        self.openDCDBtn_2.setText(_translate("MDFourier", "Open dat/dcd"))
        self.uploadbtn.setText(_translate("MDFourier", "Upload"))
        self.showBtn.setText(_translate("MDFourier", "Go Fourier"))
        self.saveBtn.setText(_translate("MDFourier", "Save data"))
        self.vmdBtn.setText(_translate("MDFourier", "Open VMD"))
        self.extBtn.setText(_translate("MDFourier", "Exit"))
        self.actionOpen.setText(_translate("MDFourier", "Open Files"))
        self.actionOpen.setIconText(_translate("MDFourier", "Open Files"))
        self.actionExport_CSV.setText(_translate("MDFourier", "Export CSV"))
        self.actionExport_All.setText(_translate("MDFourier", "Exit"))
from pyqtgraph import PlotWidget
