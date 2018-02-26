# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:25:37 2017

@author: DrLC
"""

import PIL.Image as image
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import time
import numpy
import random
import os

def load_img(img_path="Demo1.jpg"):
    
    if not os.path.exists(img_path):
        print ("ERROR: Image doesn't exist!")
        return None
    img_g = numpy.array(image.open(img_path).convert("L"))
    img = numpy.array(image.open(img_path))
    return img_g, img

def compute_grad(img):

    gx = numpy.zeros(img.shape)
    filters.sobel(img, 1, gx)    
    gy = numpy.zeros(img.shape)
    filters.sobel(img,0,gy)
    g = numpy.sqrt(gx ** 2 + gy ** 2)
    
    g_ = numpy.zeros(g.shape)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            tmp = []
            if (i > 0):
                tmp.append(abs(g[i][j]-g[i-1][j]))
            if (i < g.shape[0]-1):
                tmp.append(abs(g[i][j]-g[i+1][j]))
            if (j > 0):
                tmp.append(abs(g[i][j]-g[i][j-1]))
            if (j < g.shape[1]-1):
                tmp.append(abs(g[i][j]-g[i][j+1]))
            g_[i][j] = numpy.sum(tmp)
    
    return (img, g_)

def compute_sum(g, size=(32, 32)):
    
    if (not len(g.shape) == 2) or (not len(size) == 2):
        print ("ERROR: Wrong dimensions!")
        return None
    
    s = numpy.zeros((g.shape[0]-size[0],
                     g.shape[1]-size[1]))
    
    '''
    s[0, 0] = numpy.sum(g[0:size[0], 0:size[1]])
    for j in range(1, g.shape[1]):
        if j+size[1] >= g.shape[1]:
            continue
        s[0, j] = s[0, j-1] - numpy.sum(g[0:size[0], j-1]) \
                + numpy.sum(g[0:size[0], j-1+size[1]])
    for i in range(g.shape[0]):
        if i+size[0] >= g.shape[0]:
            continue
        for j in range(g.shape[1]):
            if j+size[1] >= g.shape[1]:
                continue
            s[i, j] = s[i-1, j] - numpy.sum(g[i-1, j:j+size[0]]) \
                + numpy.sum(g[i-1+size[0], j:j+size[1]])
    '''
    for i in range(g.shape[0]):
        if i+size[0] >= g.shape[0]:
            continue
        for j in range(g.shape[1]):
            if j+size[1] >= g.shape[1]:
                continue
            #if (not i % size[0] == 0) or (not j % size[0] == 0):
            #    s[i, j] = 1000000
            #else:
            #    s[i, j] = numpy.sum(g[i:i+size[0], j:j+size[1]])
            s[i, j] = numpy.sum(g[i:i+size[0], j:j+size[1]])
    
    return s

def get_largest(img, s, k=20, size=(32, 32)):
    
    pos = [(int(numpy.argpartition(s.reshape(numpy.prod(s.shape)), 5*k)[5*k:][i]/s.shape[1]),
                numpy.argpartition(s.reshape(numpy.prod(s.shape)), 5*k)[5*k:][i]%s.shape[1])
            for i in range(5*k)]
    
    idx = random.sample(range(len(pos)), k)
    imgs = []
    idxs = []
    for i_ in idx:
        (i, j) = pos[i_]
        idxs.append(pos[i_])
        imgs.append(img[i:i+size[0], j:j+size[1]])
    imgs = numpy.asarray(imgs)
    
    return imgs, idxs
    
def random_patch(image, k=20, size=(32, 32)):
    xs = random.sample(range(image.shape[0]-size[0]), k)
    ys = random.sample(range(image.shape[1]-size[1]), k)
    
    imgs = []
    idxs = []
    for i_ in range(k):
        idxs.append([xs[i_], ys[i_]])
        imgs.append(image[xs[i_]:xs[i_]+size[0], ys[i_]:ys[i_]+size[1]])
    imgs = numpy.asarray(imgs)
        
    return imgs, idxs
    
    
if __name__ == "__main__":
    
    t1 = time.time()
    gray_img, img = load_img("C:/Users/DrLC/Downloads/Haze/Haze-Pic/201612220930.jpg")
    size = (32, 32)
    t2 = time.time()
    print ("Load:\t%.2f sec"%(t2-t1))
    _, g = compute_grad(gray_img)
    t3 = time.time()
    print ("Grad:\t%.2f sec"%(t3-t2))
    s = compute_sum(g, size=size)
    t4 = time.time()
    print ("Sum:\t%.2f sec"%(t4-t3))
    imgs1, idx1 = get_largest(img, s, k=20, size=size)
    t5 = time.time()
    print ("Pick:\t%.2f sec"%(t5-t4))
    imgs2, idx2 = random_patch(img, k=20, size=size)
    t6 = time.time()
    print ("RP:\t%2.f sec"%(t6-t5))