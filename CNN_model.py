# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:00:16 2017

@author: DrLC
"""

import tensorflow as tf

class CNN(object):
    def __init__(self):
        tf.set_random_seed(0)
        
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.Y_ = tf.placeholder(tf.float32, [None, 6])
        self.lr = tf.placeholder(tf.float32)
        self.tst = tf.placeholder(tf.bool)
        self.iter = tf.placeholder(tf.int32)
        self.pkeep = tf.placeholder(tf.float32)
        self.pkeep_conv = tf.placeholder(tf.float32)

        def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
            exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
            bnepsilon = 1e-5
            if convolutional:
                mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
            else:
                mean, variance = tf.nn.moments(Ylogits, [0])
            update_moving_everages = exp_moving_avg.apply([mean, variance])
            m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
            v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
            Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
            return Ybn, update_moving_everages

        def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
            return Ylogits, tf.no_op()
    
        def compatible_convolutional_noise_shape(Y):
            noiseshape = tf.shape(Y)
            noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
            return noiseshape

        self.K = 16         # first convolutional layer output depth
        self.L = 32         # second convolutional layer output depth
        self.M = 64         # third convolutional layer
        self.N = 1000       # fully connected layer
        self.n_class = 6    # class label number
        
        # Convolutional kernel
        self.W1 = tf.Variable(tf.truncated_normal([6, 6, 3, self.K], stddev=0.1), name="W1")
        self.B1 = tf.Variable(tf.constant(0.1, tf.float32, [self.K]), name="B1")
        self.W2 = tf.Variable(tf.truncated_normal([5, 5, self.K, self.L], stddev=0.1), name="W2")
        self.B2 = tf.Variable(tf.constant(0.1, tf.float32, [self.L]), name="B2")
        self.W3 = tf.Variable(tf.truncated_normal([4, 4, self.L, self.M], stddev=0.1), name="W3")
        self.B3 = tf.Variable(tf.constant(0.1, tf.float32, [self.M]), name="B3")
        
        # full connection
        self.W4 = tf.Variable(tf.truncated_normal([8 * 8 * self.M, self.N], stddev=0.1), name="W4")
        self.B4 = tf.Variable(tf.constant(0.1, tf.float32, [self.N]), name="B4")
        self.W5 = tf.Variable(tf.truncated_normal([self.N, self.n_class], stddev=0.1), name="W5")
        self.B5 = tf.Variable(tf.constant(0.1, tf.float32, [self.n_class]), name="B5")
        
        stride = 1
        # now the image is 600X21
        self.XX = tf.reshape(self.X, (-1, 32, 32, 3))
        self.Y1l = tf.nn.conv2d(self.XX, self.W1, strides=[1, stride, stride, 1], padding='SAME')
        self.Y1bn, self.update_ema1 = batchnorm(self.Y1l, self.tst, self.iter, self.B1, convolutional=True)
        self.Y1r = tf.nn.relu(self.Y1bn)
        self.Y1 = tf.nn.dropout(self.Y1r, self.pkeep_conv, compatible_convolutional_noise_shape(self.Y1r))
        stride = 2  # output is 16x16
        self.Y2l = tf.nn.conv2d(self.Y1, self.W2, strides=[1, stride, stride, 1], padding='SAME')
        self.Y2bn, self.update_ema2 = batchnorm(self.Y2l, self.tst, self.iter, self.B2, convolutional=True)
        self.Y2r = tf.nn.relu(self.Y2bn)
        self.Y2 = tf.nn.dropout(self.Y2r, self.pkeep_conv, compatible_convolutional_noise_shape(self.Y2r))
        stride = 2  # output is 8x8
        self.Y3l = tf.nn.conv2d(self.Y2, self.W3, strides=[1, stride, stride, 1], padding='SAME')
        self.Y3bn, self.update_ema3 = batchnorm(self.Y3l, self.tst, self.iter, self.B3, convolutional=True)
        self.Y3r = tf.nn.relu(self.Y3bn)
        self.Y3 = tf.nn.dropout(self.Y3r, self.pkeep_conv, compatible_convolutional_noise_shape(self.Y3r))
        
        # reshape the output from the third convolution for the fully connected layer
        self.YY = tf.reshape(self.Y3, shape=[-1, 8 * 8 * self.M])
        
        self.Y4l = tf.matmul(self.YY, self.W4)
        self.Y4bn, self.update_ema4 = batchnorm(self.Y4l, self.tst, self.iter, self.B4)
        self.Y4r = tf.nn.relu(self.Y4bn)
        self.Y4 = tf.nn.dropout(self.Y4r, self.pkeep)
        self.Ylogits = tf.matmul(self.Y4, self.W5) + self.B5
        self.Y = tf.nn.softmax(self.Ylogits)

        self.update_ema = tf.group(self.update_ema1, self.update_ema2,
                                   self.update_ema3, self.update_ema4)
        
        self.cross_entropy_ = tf.nn.softmax_cross_entropy_with_logits(logits=self.Ylogits, labels=self.Y_)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy_)*100
    
        # accuracy of the trained model, between 0 (worst) and 1 (best)
        self.correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        print ('Model built!')
        
        # init
        saver = tf.train.Saver({"W1":self.W1, "B1":self.B1,
                                "W2":self.W2, "B2":self.B2,
                                "W3":self.W3, "B3":self.B3,
                                "W4":self.W4, "B4":self.B4,
                                "W5":self.W5, "B5":self.B5})

        tf_config = tf.ConfigProto()  
        tf_config.gpu_options.allow_growth = True
        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=tf_config)
        self.sess.run(init)
        saver.restore(self.sess, "model_save/model.ckpt")
    
        print ('Model loaded!')
    
    def prediction(self, x):
        label = self.sess.run(self.Y, feed_dict = {self.X: x,
                                                   self.tst: False,
                                                   self.pkeep: 1.0,
                                                   self.pkeep_conv: 1.0})
        return label