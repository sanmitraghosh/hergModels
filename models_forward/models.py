#!/usr/bin/env python2
#
# Markov model builder. To run inference for a new model just add here using the builder. NB number the model.
# 
#
from __future__ import division
from __future__ import print_function
import os
import pints
import numpy as np
from markov import MarkovModel

# Test

#     3a     2a     1a    c
# C3 --- C2 --- C1 --- O --- I
#     b      2b     3b    d
#
def Model1():
    print('Loading Model 1')
    m = MarkovModel()
    c3 = m.add_state('C3')
    c2 = m.add_state('C2')
    c1 = m.add_state('C1')
    oo = m.add_state('O', True)
    ii = m.add_state('I')
    a, b = m.add_rates('a', 'b')
    m.connect(c3, c2, a, b, 3, 1)
    m.connect(c2, c1, a, b, 2, 2)
    m.connect(c1, oo, a, b, 1, 3)
    m.connect(oo, ii, 'c', 'd')

    return m.model(), m.rate_dict, 9


"""
model.get('ikr.p'+str(1)).set_rhs(12)
print(model.code())

Model = 'Model'+str(1)
model, rate_dict_maker, dim = globals()[Model]()
rates = rate_dict_maker(np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))

model.get('ikr.p'+str(1)).set_rhs(1.2)
print(model.code())
#print(rates.iterkeys)
#keys = rates.iterkeys()
for i,k in rates.items():

    print(k[2])
"""