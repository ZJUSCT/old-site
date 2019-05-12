---
title: Distributed Tensorflow
date: 2018-12-23 19:57:25
author: 陈岩
tags: 
    - Tech
    - Tensorflow
---
# Distributed Tensorflow

## 1.单机

### log_device_placement

单机情况比较简单，不需要特殊配置，TensorFlow会自动将计算任务分配到可用的GPU上，在定义session时，可以通过*log_device_placement*参数来打印具体的计算任务分配：

```python
import tensorflow as tf

a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b

with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))
```

<img src="1.png" />

### 指定设备

如果需要让一些运算在特定的设备上执行，可以使用tf.device:

```python
import tensorflow as tf

with tf.device('/cpu:0'):
	a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
	b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/gpu:0'):
	c = a + b

with tf.Session(config = tf.ConfigProto(log_device_placement = True)) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))
```

<img src="2.png" />

### 环境变量

尽管上面一个例子中我们只给CPU和GPU0指定了计算任务，但是两块显卡的显存都被占满了：

<img src="3.png" />

因为TensorFlow会默认占满所有可见GPU的显存，对于简单的计算任务，这样显然非常浪费，我们可以通过修改环境变量*CUDA_VISIBLE_DEVICES*解决这个问题:

```shell
# 运行时指定环境变量
CUDA_VISIBLE_DEVICES=0 python demo.py
```

```python
# Python 代码中修改环境变量
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
...
```



## 2.多机

### In-graph & Between-graph

TensorFlow的分布式训练有两种模式：In-graph和Between-graph

In-graph: 不同的机器执行计算图的不同部分，和单机多GPU模式类似，一个节点负责模型数据分发，其他节点等待接受任务，通过*tf.device("/job:worker/task:n")*来指定计算运行的节点

<img src="5.png" />

Between-graph:每台机器执行相同的计算图

 > Author: 陈岩
 > PostDate: 2018.12.21
