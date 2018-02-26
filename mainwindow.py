# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 620)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image_label = QtWidgets.QLabel(self.centralwidget)
        self.image_label.setGeometry(QtCore.QRect(20, 100, 560, 480))
        self.image_label.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.image_label.setAutoFillBackground(True)
        self.image_label.setText("")
        self.image_label.setObjectName("image_label")
        self.patch_button = QtWidgets.QPushButton(self.centralwidget)
        self.patch_button.setGeometry(QtCore.QRect(400, 20, 80, 40))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.patch_button.setFont(font)
        self.patch_button.setObjectName("patch_button")
        self.path_button = QtWidgets.QPushButton(self.centralwidget)
        self.path_button.setGeometry(QtCore.QRect(300, 20, 80, 40))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.path_button.setFont(font)
        self.path_button.setObjectName("path_button")
        self.path_line = QtWidgets.QLineEdit(self.centralwidget)
        self.path_line.setGeometry(QtCore.QRect(20, 20, 261, 40))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(8)
        self.path_line.setFont(font)
        self.path_line.setReadOnly(True)
        self.path_line.setObjectName("path_line")
        self.go_button = QtWidgets.QPushButton(self.centralwidget)
        self.go_button.setGeometry(QtCore.QRect(500, 20, 80, 40))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.go_button.setFont(font)
        self.go_button.setObjectName("go_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 17))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Air Quality Prediction"))
        self.patch_button.setText(_translate("MainWindow", "PATCH"))
        self.path_button.setText(_translate("MainWindow", "PATH"))
        self.go_button.setText(_translate("MainWindow", "GO!"))

