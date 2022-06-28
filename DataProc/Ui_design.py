# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\baranov_ma\Documents\myProjects\MD\DataProc\design.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PacketProcessor(object):
    def setupUi(self, PacketProcessor):
        PacketProcessor.setObjectName("PacketProcessor")
        PacketProcessor.resize(493, 541)
        self.centralwidget = QtWidgets.QWidget(PacketProcessor)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setStyleSheet("background-color: rgb(235, 217, 255);")
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 9)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setObjectName("widget_2")
        self.gridLayout = QtWidgets.QGridLayout(self.widget_2)
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView = PlotWidget(self.widget_2)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.widget_3 = QtWidgets.QWidget(self.widget)
        self.widget_3.setMaximumSize(QtCore.QSize(16777215, 41))
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.path_label = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(9)
        self.path_label.setFont(font)
        self.path_label.setText("")
        self.path_label.setObjectName("path_label")
        self.horizontalLayout_2.addWidget(self.path_label)
        self.verticalLayout_2.addWidget(self.widget_3)
        self.widget_4 = QtWidgets.QWidget(self.widget)
        self.widget_4.setMaximumSize(QtCore.QSize(16777215, 41))
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.changeDir_btn = QtWidgets.QPushButton(self.widget_4)
        self.changeDir_btn.setMinimumSize(QtCore.QSize(90, 28))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(9)
        self.changeDir_btn.setFont(font)
        self.changeDir_btn.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(199, 205, 255);\n"
"}")
        self.changeDir_btn.setObjectName("changeDir_btn")
        self.horizontalLayout.addWidget(self.changeDir_btn)
        self.process_btn = QtWidgets.QPushButton(self.widget_4)
        self.process_btn.setMinimumSize(QtCore.QSize(85, 28))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(9)
        self.process_btn.setFont(font)
        self.process_btn.setStyleSheet("QPushButton {\n"
"    border:none;\n"
"    border-radius: 8px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    background-color: rgb(199, 205, 255);\n"
"}")
        self.process_btn.setObjectName("process_btn")
        self.horizontalLayout.addWidget(self.process_btn)
        self.verticalLayout_2.addWidget(self.widget_4)
        self.verticalLayout.addWidget(self.widget)
        PacketProcessor.setCentralWidget(self.centralwidget)

        self.retranslateUi(PacketProcessor)
        QtCore.QMetaObject.connectSlotsByName(PacketProcessor)

    def retranslateUi(self, PacketProcessor):
        _translate = QtCore.QCoreApplication.translate
        PacketProcessor.setWindowTitle(_translate("PacketProcessor", "Packet Processor"))
        self.changeDir_btn.setText(_translate("PacketProcessor", "Выбрать папку"))
        self.process_btn.setText(_translate("PacketProcessor", "Обработать"))
from pyqtgraph import PlotWidget