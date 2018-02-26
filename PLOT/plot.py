# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:17:05 2017

@author: DrLC
"""

import pickle, gzip
import numpy
import matplotlib.pyplot as plt



with gzip.open("HazeCNN_fft_curve_log.pkl.gz", 'rb') as f:
    p = pickle.load(f)

q = []
for j in range(len(p)):
    q.append([numpy.mean(p[j][i*10:(i+1)*10]) for i in range((int)(len(p[j])/10))])

plt.subplot(2, 1, 1)    
plt.title("CNN on Haze-Pic Dataset with FFT Features")
plt.plot(p[2], p[3], color=(1, 0, 0, 0.3), label="Train Curve")
plt.plot(p[0], p[1], color=(0, 0, 1, 0.5), label="Test Curve")
plt.ylim(0, 1.0)
plt.legend(loc="lower right", fontsize='small', ncol=2)
plt.subplot(2, 1, 2)
plt.plot(q[2], q[3], color=(1, 0, 0, 0.5), label="Train Curve")
plt.plot(q[0], q[1], color=(0, 0, 1, 0.5), label="Test Curve")
test_acc = numpy.mean(p[1][int(len(p[1])/10)*9:])
train_acc = numpy.mean(p[3][int(len(p[3])/10)*9:])
plt.plot([0, p[2][-1]], [train_acc, train_acc], "--",
         color=(1, 0, 0, 0.1), label="Train acc = %.1f%%"%(100*train_acc))
plt.plot([0, p[0][-1]], [test_acc, test_acc], "--",
         color=(0, 0, 1, 0.1), label="Test acc = %.1f%%"%(100*test_acc))
plt.ylim(0, 1.0)
plt.legend(loc="upper right", fontsize='small', ncol=2)
plt.savefig("CNN_figure.jpg")
plt.show()