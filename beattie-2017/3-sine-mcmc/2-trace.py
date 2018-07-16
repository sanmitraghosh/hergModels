#!/usr/bin/env python
#
# Show traces of generated chains
#
import os
import sys
import numpy as np
import pints
import pints.io
import pints.plot
import matplotlib.pyplot as plt

# Parse command line arguments
if len(sys.argv) != 3:
    print('Usage:  2-trace.py <nchains> <filename>')
    sys.exit(1)
chains = int(sys.argv[1])
filename = sys.argv[2]

# Load chains
chains = pints.io.load_samples(filename, chains)

pints.plot.trace(*chains)

plt.show()
