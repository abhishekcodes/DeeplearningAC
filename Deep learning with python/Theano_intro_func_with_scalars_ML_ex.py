# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:19:45 2019

@author: Durgesh
"""

import theano.tensor as T
from theano import function

a = T.dscalar('a')
b = T.dscalar('b')
c = T.dscalar('c')
d = T.dscalar('d')
e = T.dscalar('e')

f = ((a-b+c)*d)/e

g = function([a,b,c,d,e],f)

print("Expected : ((1-2+3)*4)/5 =", ((1-2+3)*4)/5)
print("Via Theano: ",g(1,2,3,4,5))
