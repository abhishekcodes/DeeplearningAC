# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:34:21 2019

@author: Durgesh
"""

import theano.tensor as T
from theano import function

#sigmoid
a = T.dmatrix('a')
f_a = T.nnet.sigmoid(a)
f_sigmoid = function([a],[f_a])
print("sigmoid:", f_sigmoid([[-1,0,1]]))

#tanh
b= T.dmatrix('b')
f_b= T.tanh(b)
f_tanh = function([b],[f_b])
print("tanh:", f_tanh([[-1,0,1]]))

#fast sigmoid
c = T.dmatrix('c')
f_c= T.nnet.ultra_fast_sigmoid(c)
f_fast_sigmoid =function([c],[f_c])
print("Fast sigmoid:", f_fast_sigmoid([[-1,0,1]]))

#softplus
d = T.dmatrix('d')
f_d = T.nnet.softplus('d')
f_softplus= function([d],[f_d])
print("Soft Plus:", f_softplus([[-1,0,1]]))

#relu
e = T.dmatrix('e')
f_e = T.nnet.relu(e)
f_relu = function([e],[f_e])
print("relu:", f_relu([[-1,0,1]]))

#softmax
f = T.dmatrix('f')
f_f = T.nnet.softmax(f)
f_softmax = function([f],[f_f])
print("softmax:",f_softmax([[-1,0,1]]))

























