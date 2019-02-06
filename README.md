# herg_testing

## Gary's notes

This is the rough order of operation.

1. Run `cmaesfit.py` with various options to find some good optimisation start points. The noise model includes 1) iid noise 2) independant but not identically distributed Gaussian noise. Here are the basic options to run this code:
 In iid mode use the following basic command:
 
`python2 cmaesfit.py` 

In non iid (discrepancy mode) use this:

`python2 cmaesfit.py --discrepancy True`
 
 By default we use a log-linear transform, and this can be changed using the option `--transform`, see the `cmaesfit.py` file for more details. NB: the discrepancy noise parameters are not transformed for any choice of transforming the ion channel parameters.
 TODO: Figure out good transform for the discrepancy parameters.
 
2. [Optional] Run `sinemcmc.py` to run MCMC chains from the best locations. THIS CAN TAKE A WHILE (days!). As the CMA-ES optimisation this also has two noise models, so either run:
For iid noise:

`python2 sinemcmc.py`

Or for discrepancy run:

`python2 sinemcmc.py --discrepancy True`

By default the MCMC will initialise the chains from CMA-ES fit of the iid noise model, but if you want to initialise the chains from CMA-ES fit of a discrepancy model then 1) make sure that you have run CMA-ES in the discrepancy mode and 2) Use the following command:

`python2 sinemcmc.py --discrepancy True --init_ds True`

Finally, if you want to calculate the marginal likelihood using thermodynamic integration then use the following command, NB this is not supported for discrepancy model (non-iid noise) at the present:

`python2 sinemcmc.py --thermo True`

The above command runs MCMC chains at different temperatures to sample from power posteriors. You can control the number of temperatures (NB increasing this will increase run time) with the option `--ntemps`. See the file `sinemcmc.py` file for other options.

3. Run `make_predictions.py` which will print out the likelihood of the best sine wave fits seen in either CMA-ES or MCMC, and the action potential predictions for these parameter sets.
