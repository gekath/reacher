from utils import *
from function_names import *
from numpy import *
import tensorflow as tf
import numpy as np

class Net():

    def __init__(self):
        pass

    def train(self,input,out,nhid,val_in,val_out):
        keep_prob = tf.placeholder(tf.float32)
        momentum = tf.placeholder(tf.float32)

        # tf.nn.l2_normalize()

        xDim = input.shape[1]
        yDim = out.shape[1]
        sess = tf.InteractiveSession()
        x = tf.placeholder(tf.float32, shape=[None, xDim])
        dx = xDim
        dh = nhid
        dy = yDim
        y_ = tf.placeholder(tf.float32, shape=[None, yDim])
        Wx = tf.Variable(tf.random_normal([dx, dh]))
        Bx = tf.Variable(tf.random_normal([dh]))
        Wh = tf.Variable(tf.random_normal([dh, dy]))
        Bh = tf.Variable(tf.random_normal([dy]))

        # Wx = tf.Print(Wx, [Wx],"Weight: ")
        # Bx = tf.Print(Bx, [Bx], "Bias: ")
        # x = tf.Print(x, [x], "X: ")
        # matmul_result = tf.matmul(x, Wx)
        # matmul_result = tf.Print(matmul_result, [matmul_result], "Matmul Result: ")
        # y = tf.nn.softmax(matmul_result + Bx)
        # raw_input()

        h = tf.nn.softmax(tf.matmul(x, Wx) + Bx)
        #y = tf.nn.xw_plus_b(x, Wx, Bx)#tf.nn.tanh(tf.matmul(x, Wx) + Bx) #tf.nn.xw_plus_b(x, Wx, Bx)#tf.nn.tanh(tf.matmul(x, Wx) + Bx)  #tf.nn.xw_plus_b(x, Wx, Bx)# # Consider replacing with xw_plus_b
        h_fc1_drop = tf.nn.dropout(h,keep_prob)
        y = tf.nn.softmax(tf.matmul(h_fc1_drop, Wh) + Bh)
        regularizers = tf.nn.l2_loss(Wx) + tf.nn.l2_loss(Bx) + tf.nn.l2_loss(Wh) + tf.nn.l2_loss(Bh)

        loss = tf.reduce_mean((y_ - y)**2) + .001*regularizers
        train_step = tf.train.MomentumOptimizer(.5, momentum).minimize(loss)

        sess.run(tf.initialize_all_variables())

        mom = .95
        prev_acc = Inf
        for i in range(100):
            train_step.run(feed_dict={x: input, y_: out, keep_prob:.5, momentum:mom})
            pred = sess.run([y],feed_dict={x: val_in, y_: val_out, keep_prob:1, momentum:mom})
            pred = np.asarray(pred,dtype=float32)[0,:,:]
            # print(pred)
            # print(pred.shape)
            acc = np.sum((pred - val_out)**2)
            if (i + 1) % 20 == 0:
                print("Error at step {} is {}".format(i + 1, acc))
            # print(acc)
            # if ((acc-prev_acc)[0] > .1):
            #     print('meow')
            #prev_acc = acc
            # print(np.mean(acc))
        return sess, y, x, keep_prob, momentum

    def train_2_hidden(self, input, out, nhid, val_in, val_out):
        keep_prob = tf.placeholder(tf.float32)
        momentum = tf.placeholder(tf.float32)

        # tf.nn.l2_normalize()

        xDim = input.shape[1]
        yDim = out.shape[1]
        sess = tf.InteractiveSession()
        x = tf.placeholder(tf.float32, shape=[None, xDim])
        dx = xDim
        dh1 = nhid
        dh2 = nhid-20
        dy = yDim
        y_ = tf.placeholder(tf.float32, shape=[None, yDim])
        Wx = tf.Variable(tf.random_normal([dx, dh1]))
        Bx = tf.Variable(tf.random_normal([dh1]))
        Wh1 = tf.Variable(tf.random_normal([dh1,dh2]))
        Bh1 = tf.Variable(tf.random_normal([dh2]))
        Wh2 = tf.Variable(tf.random_normal([dh2, dy]))
        Bh2 = tf.Variable(tf.random_normal([dy]))

        # Wx = tf.Print(Wx, [Wx],"Weight: ")
        # Bx = tf.Print(Bx, [Bx], "Bias: ")
        # x = tf.Print(x, [x], "X: ")
        # matmul_result = tf.matmul(x, Wx)
        # matmul_result = tf.Print(matmul_result, [matmul_result], "Matmul Result: ")
        # y = tf.nn.softmax(matmul_result + Bx)
        # raw_input()

        h1 = tf.nn.softmax(tf.matmul(x, Wx) + Bx)
        # y = tf.nn.xw_plus_b(x, Wx, Bx)#tf.nn.tanh(tf.matmul(x, Wx) + Bx) #tf.nn.xw_plus_b(x, Wx, Bx)#tf.nn.tanh(tf.matmul(x, Wx) + Bx)  #tf.nn.xw_plus_b(x, Wx, Bx)# # Consider replacing with xw_plus_b
        h_fc1_drop = tf.nn.dropout(h1, keep_prob)
        h2 = tf.nn.softmax(tf.matmul(h_fc1_drop, Wh1) + Bh1)
        h2_fc1_drop = tf.nn.dropout(h2, keep_prob)
        y = tf.nn.softmax(tf.matmul(h2_fc1_drop, Wh2) + Bh2)
        regularizers = tf.nn.l2_loss(Wx) + tf.nn.l2_loss(Bx) + tf.nn.l2_loss(
            Wh1) + tf.nn.l2_loss(Bh1) + tf.nn.l2_loss(Wh2) + tf.nn.l2_loss(Bh2)

        loss = tf.reduce_mean((y_ - y) ** 2) + .001 * regularizers
        train_step = tf.train.GradientDescentOptimizer(.01).minimize(loss)
        # train_step = tf.train.MomentumOptimizer(.5, momentum).minimize(
        #     loss)

        sess.run(tf.initialize_all_variables())

        mom = .95
        prev_acc = Inf
        for i in range(100):
            train_step.run(
                feed_dict={x: input, y_: out, keep_prob: .5, momentum: mom})
            pred = sess.run([y],
                            feed_dict={x: val_in, y_: val_out, keep_prob: 1,
                                       momentum: mom})
            pred = np.asarray(pred, dtype=float32)[0, :, :]
            # print(pred)
            # print(pred.shape)
            # print(pred)
            # print(val_out)
            acc = np.sum((pred - val_out) ** 2)
            print(acc)
            if (i + 1) % 20 == 0:
                print("Error at step {} is {}".format(i + 1, acc))
                # print(acc)
                # if ((acc-prev_acc)[0] > .1):
                #     print('meow')
                # prev_acc = acc
                # print(np.mean(acc))
        return sess, y, x, keep_prob, momentum