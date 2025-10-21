'''
Oct 20 2025

First attempt at creating a basis function to represent the visual servoing function.
We can first try with many different basis functions and analyze the coefficients to determine 
which functions are effective bases.


'''
from ..robot_toolbox.create_uvs import UVS
import numpy as np
import sympy as sp
import logging

