# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as t
import numpy as np              #numpy，数学工具库

#create data
x_data = np.random.rand(100).astype(np.float32)  #flaot32 是一种常用的数据类型
y_data = x_data*0.1+0.3                   #pay attention to '0.1' &. '0.3'

###create tensorflow structure start###
Weight=t.Variable(t.random_uniform([1],-1.0,1.0))     #Variable 位置参数变量
biases=t.Variable(t.zeros([1]))                       #用weight和biases去拟合0.1与0.3
                                                      #
y = Weight*x_data+biases
               
#optimizer，优化器*KEY*#
loss=t.reduce_mean(t.square(y-y_data))  #caculate the loss
optimizer = t.train.GradientDescentOptimizer(0.5)   #反向传播误差，采用梯度下降法，

#训练目标#
train = optimizer.minimize(loss)
#初始化#
init = t.initialize_all_variables()
###create tensorflow structure end ###

sess = t.Session()          #
sess.run(init)              # initialize

for step in range(201):
    sess.run(train)
    if step % 5 == 0:
        print(step,sess.run(Weight),sess.run(biases))