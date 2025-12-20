#!/usr/bin/env python
'''
Dec 19 2205

Using the Jacobian basis object
'''


from sklearn.preprocessing import PolynomialFeatures
from robot_toolbox.create_uvs import UVS
import matplotlib.pyplot as plt

import numpy as np
import sympy as sp
import logging

import os
from datetime import datetime
import sys

# Configure basic logging to the console (default level is WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for the current module
# logger. debug, info, warning, error, critical
