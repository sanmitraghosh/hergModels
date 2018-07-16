#!/usr/bin/env python
#
# Show generated chains
#
import os
import sys
import numpy as np
import pints.io
import pints.plot
import matplotlib.pyplot as plt

# Parse command line arguments
if len(sys.argv) != 2:
    print('Usage:  4-pairwise.py <filename>')
    sys.exit(1)
filename = sys.argv[1]

# Load chain
chain = pints.io.load_samples(filename)

# Remove burn-in
chain = chain[50000:]

# Apply thinning
chain = chain[::10]

pints.plot.pairwise(chain, opacity=0.05)

plt.show()
