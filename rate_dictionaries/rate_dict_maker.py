from __future__ import division
from __future__ import print_function
import cPickle
import os

model_name = 'model-0'
rate0 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'] ,
3: [int(4), int(5), 'positive'] ,
4: [int(6), int(7), 'negative']

}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate0, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-2'
rate2 = {
1: [0.0007776874544055468, 0.0010996899009535906, 'positive'],
2: [0.00036800640630228824, 0.027311983503556143, 'negative'] ,
3: [0.09735309667134465, 0.014083571771890785, 'positive'] ,
4: [0.0043111538424118094, 0.03437923245146323, 'negative']

}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate2, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-3'
rate3 = {
1: [int(0), int(1), 'positive'],
2: [int(2), int(3), 'negative'] ,
3: [int(4), int(5), 'positive'] ,
4: [int(6), int(7), 'negative']
}

root = os.path.abspath('rate_dictionaries')
rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate3, open(rate_filename, 'wb')) 
#################################################################################
model_name = 'model-16'
rate16 = {
1: [0.0007776874544055468, 0.0010996899009535906, 'positive'],
2: [0.00036800640630228824, 0.027311983503556143, 'negative'] ,
3: [0.09735309667134465, 0.014083571771890785, 'positive'] ,
4: [0.0043111538424118094, 0.03437923245146323, 'negative'],
5: [0.09735309667134465, 0.014083571771890785, 'positive'] ,
6: [0.0043111538424118094, 0.03437923245146323, 'negative'],
7: [0.09735309667134465, 0.0, 'vol_ind'] ,
8: [0.0043111538424118094, 0.0,'vol_ind']
}

root = os.path.abspath('rate_dictionaries')

rate_filename = os.path.join(root, model_name + '-priors.p')
cPickle.dump(rate16, open(rate_filename, 'wb')) 