from __future__ import division
from __future__ import print_function
import cPickle
import os

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
