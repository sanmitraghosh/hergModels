#!/usr/bin/env python3
import os
import sys
import numpy as np
import pints
import pints.io
import cPickle
import models_forward.util as util
import models_forward.Rates as Rates



num_chains = 5 # HARDCODED. 
transform=1 # HARDCODED. 
cell = 5 # HARDCODED

folder = 'mcmc_results'
for model in range(1,30):

    model_name = 'model-'+str(model)
    root = os.path.abspath('rate_dictionaries')
    rate_file = os.path.join(root, model_name + '-priors.p')
    rate_dict = cPickle.load(open(rate_file, 'rb'))

    LL_filename = folder + '/' + model_name  + '-cell-' + str(cell) + '-LLs.csv'
    if not os.path.isfile(LL_filename):
        continue

    LLs = pints.io.load_samples(LL_filename)
    chains = pints.io.load_samples(folder + '/' + model_name  + '-cell-' + str(cell) + '-chain.csv', n=num_chains)

    # Set parameter transformation
    #transform_to_model_param = parametertransform.log_transform_to_model_param
    #transform_from_model_param = parametertransform.log_transform_from_model_param

    MAPs = np.argmax(LLs, axis=0)
    which_chain = np.argmax([LLs[MAPs[i]][i] for i in range(len(MAPs))])

    transform_MAP_param = chains[which_chain][MAPs[which_chain], :]
    MAP_param = util.transformer(transform, transform_MAP_param, rate_dict, False)
    print(model_name,': Best Log-Posterior: ', LLs[MAPs[which_chain]][which_chain])
    print('MAP parameters: ')
    print(MAP_param)
    #from defaultsetting import param_names
    with open('mcmc_results/' + model_name + '-cell-' + str(cell) + '-best-parameters.txt', 'w') as f:
        for p in transform_MAP_param:
            f.write(pints.strfloat(p) + '\n')
