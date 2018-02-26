# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 21:46:13 2017

@author: DrLC
"""

import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
import mainwindow
import Cal_Grad
import CNN_model
import numpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def plot_bar(res_, path):
    bar_width = 0.5
    #res_[0] = res_[0]/3*2
    res = numpy.asarray(res_) / numpy.sum(res_)
    plt.bar([0.5], [res[0]],
            bar_width, color=(1/6,1/6,1/6,0.5), label='')
    plt.bar([1.5], [res[1]],
            bar_width, color=(2/6,2/6,2/6,0.5), label='')
    plt.bar([2.5], [res[2]],
            bar_width, color=(3/6,3/6,3/6,0.5), label='')
    plt.bar([3.5], [res[3]],
            bar_width, color=(4/6,4/6,4/6,0.5), label='')
    plt.bar([4.5], [res[4]],
            bar_width, color=(5/6,5/6,5/6,0.5), label='')
    plt.bar([5.5], [res[5]],
            bar_width, color=(6/6,6/6,6/6,0.5), label='')
    plt.xlabel("Air Quality")
    plt.ylabel("Probability")
    plt.xticks((0.75, 1.75, 2.75, 3.75, 4.75, 5.75),
               ('1', '2', '3', '4', '5', '6'))
    plt.savefig(path)
    plt.show()
    return res

class UI(object):
    
    def path_button_click(self):
        '''
        path_button clicked handler
        '''
        print ("'PATH' button clicked!")
        dialog = QtWidgets.QFileDialog()
        fname=dialog.getOpenFileName(caption="Choose your image",
                                     filter="JPEG Files(*.jpg);;PNG Files(*.png)")
        print (fname[0], fname[1])
        self.fname = fname[0]
        self.ui.path_line.setText(fname[0])
        image = QtGui.QImage()
        image.load(fname[0])
        self.ui.image_label.setPixmap(QtGui.QPixmap.fromImage(image))
    
    def patch_button_click(self):
        '''
        go_button clicked handler
        '''
        print ("'PATCH' button clicked!")
        if (self.fname is None) or (not os.path.exists(self.fname)):
            print ("\tError: Wrong path!")
            self.ui.path_line.setText("Error: Wrong path --" + self.fname)
        
        gray_img, img = Cal_Grad.load_img(self.fname)
        print ("\tImage loaded!")
        _, grad = Cal_Grad.compute_grad(gray_img)
        print ("\tGradient calculated!")
        s = Cal_Grad.compute_sum(grad, size=self.patch_size)
        print ("\tSummation calculated!")
        imgs1, idxs1 = Cal_Grad.get_largest(img, s, k=200, size=self.patch_size)
        print ("\tThe largest picked!")
        for (i, j) in idxs1:
            for i_ in range(i, i+self.patch_size[0]):
                img[i_][j][0] = 255
                img[i_][j][1] = 0
                img[i_][j][2] = 0
                img[i_][j+self.patch_size[1]-1][0] = 255
                img[i_][j+self.patch_size[1]-1][1] = 0
                img[i_][j+self.patch_size[1]-1][2] = 0
            for j_ in range(j, j+self.patch_size[1]):
                img[i][j_][0] = 255
                img[i][j_][1] = 0
                img[i][j_][2] = 0
                img[i+self.patch_size[0]-1][j_][0] = 255
                img[i+self.patch_size[0]-1][j_][1] = 0
                img[i+self.patch_size[0]-1][j_][2] = 0
        self.imgs = imgs1
        print ("\tImage altered!")
        mpimg.imsave("tmp/tmpimg.jpg", img)
        image = QtGui.QImage()
        image.load("tmp/tmpimg.jpg")
        self.ui.image_label.setPixmap(QtGui.QPixmap.fromImage(image))
        print ("\tAll done!")
    
    def go_button_click(self):
        print ("'GO!' button clicked!")
        if self.imgs is None:
            print ("\tError: No image loaded!")
            return
        res1 = self.model.prediction(numpy.asarray(self.imgs,
                                                   dtype="float32"))
        res = [0, 0, 0, 0, 0, 0]
        for i in res1:
            res[numpy.argmax(i)] += 1
        print ("\tVote:", end=" ")
        print (res)
        prob = plot_bar(res, "tmpbarimg.jpg")
        print ("\tBar image generated!")

        dialog = QtWidgets.QDialog()
        dialog.setFixedSize(500, 500)
        
        showstr = "Air Quality --\n"
        prob = list(prob)
        aqi = numpy.argmax(prob)
        if (aqi == 0):
            showstr += " GOOD (0-35) !    "
        elif (aqi == 1):
            showstr += " FINE (35-75) !    "
        elif (aqi == 2):
            showstr += " SLIGHT POLLUTION (75-115) !    "
        elif (aqi == 3):
            showstr += " MEDIUM POLLUTION (115-150) !    "
        elif (aqi == 4):
            showstr += " HEAVY POLLUTION (150-250) !    "
        elif (aqi == 5):
            showstr += " SEVERE POLLUTION (250-500) !    "
        print ("\t"+showstr)

        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(10)
        font.setBold(True)
        text_label = QtWidgets.QLabel(dialog)
        text_label.setFont(font)
        text_label.setGeometry(QtCore.QRect(20, 20, 460, 170))
        text_label.setText(showstr)
        
        bar_label = QtWidgets.QLabel(dialog)
        bar_label.setGeometry(QtCore.QRect(70, 210, 360, 270))
        bar_label.setScaledContents(True)
        image = QtGui.QImage()
        image.load("tmpbarimg.jpg")
        bar_label.setPixmap(QtGui.QPixmap.fromImage(image))
        
        dialog.exec_()
            
        
    def simple_patch_button_click(self):
        print ("'PATCH' button clicked!")
        if (self.fname is None) or (not os.path.exists(self.fname)):
            print ("Error: Wrong path!")
            self.ui.path_line.setText("Error: Wrong path --" + self.fname)
        
        img = numpy.asarray(mpimg.imread(self.fname))
        imgs = []
        for i in range(int(img.shape[0]/self.patch_size[0])):
            for j in range(int(img.shape[1]/self.patch_size[1])):
                imgs.append(img[i*self.patch_size[0]:(i+1)*self.patch_size[0],
                                j*self.patch_size[1]:(j+1)*self.patch_size[1]])
        imgs = numpy.asarray(imgs)
        self.imgs = imgs
        print ("\tImage loaded!")
        print ("\tshape=",imgs.shape)
    
    def simple_go_button_click(self):
        print ("'GO!' button clicked!")
        res = self.model.prediction(numpy.asarray(self.imgs,
                                                  dtype="float32"))
        res = numpy.sum(res, 0)
        print ("\tVote:", res)
        prob = plot_bar(res, "tmpbarimg.jpg")
        
        
    def __init__(self):
        '''
        Initialize
        '''
        self.fname = None
        self.imgs = None
        self.model = CNN_model.CNN()
        self.patch_size = (32, 32)
        self.app = QtWidgets.QApplication(sys.argv)
        self.MainWindow = QtWidgets.QMainWindow()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self.MainWindow)
        self.MainWindow.setFixedSize(self.MainWindow.width(), self.MainWindow.height())
        self.ui.image_label.setScaledContents(True)
        self.ui.image_label.setMargin(5)
        self.ui.path_button.clicked.connect(self.path_button_click)
        self.ui.patch_button.clicked.connect(self.patch_button_click)
        #self.ui.patch_button.clicked.connect(self.simple_patch_button_click)
        self.ui.go_button.clicked.connect(self.go_button_click)
        #self.ui.go_button.clicked.connect(self.simple_go_button_click)
        
if __name__ == "__main__":
    ui = UI()
    ui.MainWindow.show()
    sys.exit(ui.app.exec_())