# herg_testing

## Gary's notes

This is the rough order of operation.

1. Run `cmaesfit.py` with various options to find some good optimisation start points. You can re-run and it will not overwrite better guesses if any are recorded.
2. [Optional] Run `sinemcmc.py` to run MCMC chains from the best locations. THIS CAN TAKE A WHILE (days!).
3. Run `make_predictions.py` which will print out the likelihood of the best sine wave fits seen in either CMA-ES or MCMC, and the action potential predictions for these parameter sets.