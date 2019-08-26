# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:48:01 2019

@author: Durgesh
"""

import numpy
import theano.tensor as T
from theano import function

a = T.dmatrix('a')
b = T.dmatrix('b')
c = T.dmatrix('c')
d = T.dmatrix('d')

e = (a+b-c)*d

f = function([a,b,c,d],e)

a_data = numpy.array(([1,1],[1,1]))
b_data = numpy.array(([2,2],[2,2]))
c_data = numpy.array(([5,5],[5,5]))
d_data = numpy.array(([3,3],[3,3]))

print("expected:", (a_data+b_data- c_data)*d_data)
print("via Theano:", f(a_data,b_data,c_data,d_data))
