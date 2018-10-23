from __future__ import division
from __future__ import print_function
import cPickle
import os
# Gary says:
# This file creates some 'rate dictionaries' that simply state whether parameters
# are involved in 'positive' rates (increase with increasing Voltage); or 
# 'negative' rates (increase with decreasing voltage). 
#
# i.e. rate=A*exp(BV) is called 'positive'; rate=A*exp(-BV) is called 'negative'.
# A special case is 'vol_ind' which signifies a voltage independent rate 
# (rate=A with B hardcoded to zero).
#


#################################################################################
model_name = 'model-0'
rate0 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative']

}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate0, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-1'
rate1 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate1, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-2'
rate2 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate2, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-3'
rate3 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate3, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-4'
rate4 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate4, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-5'
rate5 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate5, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-6'
rate6 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'],
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate6, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-7'
rate7 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate7, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-8'
rate8 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'],
6: [int(10), int(11), 'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate8, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-9'
rate9 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate9, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-10'
rate10 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'],
6: [int(10), int(11), 'negative']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate10, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-11'
rate11 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'],
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'positive']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate11, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-12'
rate12 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'],
6: [int(10), int(11), 'positive']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate12, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-13'
rate13 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'],
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'positive'],
9: [int(16), int(17), 'positive'],
10: [int(18), int(19), 'negative'],
11: [int(20), int(21), 'positive'],
12: [int(22), int(23), 'negative'],
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate13, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-14'
rate14 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate14, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-15'
rate15 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3),  'negative'],
3: [int(4), int(5),  'positive'],
4: [int(6), int(7),  'negative'],
5: [int(8), int(9),  'positive'],
6: [int(10), int(11),  'negative'],
7: [int(12), int(13),  'positive'],
8: [int(14), int(15),  'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate15, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-16'
rate16 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3),  'negative'],
3: [int(4), int(5),  'positive'],
4: [int(6), int(7),  'negative'],
5: [int(8), int(9),  'positive'] ,
6: [int(10), int(11),  'negative'],
7: [int(12), 0.0, 'vol_ind'],
8: [int(13), 0.0,'vol_ind']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate16, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-17'
rate17 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3),  'negative'],
3: [int(4), int(5),  'positive'],
4: [int(6), int(7),  'negative'],
5: [int(8), int(9),  'positive']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate17, open(rate_filename, 'wb')) 

#################################################################################
model_name = 'model-18'
rate18 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3),  'negative'],
3: [int(4), int(5),  'positive'],
4: [int(6), int(7),  'negative'],
5: [int(8), int(9),  'positive'],
6: [int(10), int(11),  'negative'],
7: [int(12), int(13),  'positive'],
8: [int(14), int(15),  'negative'],
9: [int(16), int(17),  'positive']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate18, open(rate_filename, 'wb')) 

#################################################################################
model_name = 'model-19'
rate19 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate19, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-20'
rate20 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'] ,
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'negative']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate20, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-21'
rate21 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'] ,
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'negative'],
9: [int(16), int(17), 'positive'],
10: [int(18), int(19), 'positive'],
11: [int(20), int(21), 'positive']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate21, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-22'
rate22 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'] ,
6: [int(10), int(11), 'positive'],
7: [int(12), int(13), 'positive']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate22, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-23'
rate23 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'] ,
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'negative'],
9: [int(16), int(17), 'positive'],
10: [int(18), int(19), 'negative'],
11: [int(20), int(21), 'positive'],
12: [int(22), int(23), 'negative'],
13: [int(24), int(25), 'positive'],
14: [int(26), int(27), 'negative'],
15: [int(28), int(29), 'positive'],
16: [int(30), int(31), 'positive'],
17: [int(32), int(33), 'positive']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate23, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-24'
rate24 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate24, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-25'
rate25 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'] ,
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'negative'],
9: [int(16), int(17), 'positive'],
10: [int(18), int(19), 'negative']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate25, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-26'
rate26 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate26, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-27'
rate27 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'] ,
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'negative'],
9: [int(16), int(17), 'positive'],
10: [int(18), int(19), 'negative']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate27, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-28'
rate28 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'] ,
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'negative'],
9: [int(16), int(17), 'positive'],
10: [int(18), int(19), 'positive'],
11: [int(20), int(21), 'positive'],
12: [int(22), int(23), 'positive'],
13: [int(24), int(25), 'negative'],
14: [int(26), int(27), 'positive']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate28, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-29'
rate29 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'] ,
6: [int(10), int(11), 'positive'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'positive']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate29, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-30'
rate30 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'],
3: [int(4), int(5), 'positive'],
4: [int(6), int(7), 'negative'],
5: [int(8), int(9), 'positive'] ,
6: [int(10), int(11), 'negative'],
7: [int(12), int(13), 'positive'],
8: [int(14), int(15), 'negative'],
9: [int(16), int(17), 'positive'],
10: [int(18), int(19), 'negative'],
11: [int(20), int(21), 'positive'],
12: [int(22), int(23), 'negative'],
13: [int(24), int(25), 'positive'],
14: [int(26), int(27), 'negative'],
15: [int(28), int(29), 'positive'],
16: [int(30), int(31), 'positive'],
17: [int(32), int(33), 'positive'],
18: [int(34), int(35), 'positive'],
19: [int(36), int(37), 'negative'],
20: [int(38), int(39), 'positive'],
21: [int(40), int(41), 'negative'],
22: [int(42), int(43), 'positive'],
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate30, open(rate_filename, 'wb')) 
#################################################################################
