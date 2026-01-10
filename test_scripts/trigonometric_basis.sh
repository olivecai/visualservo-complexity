#!/bin/bash

python3 ./src/potential_basis.py dof3 0,1 50 3 500 10 1,2,3,4 0,1 88 example_dof3_jan7
python3 ./src/potential_basis.py puma 0,1 50 3 500 10 1,2,3,4,5 0,1 88 example_puma_jan7
python3 ./src/potential_basis.py jaco 0,1 50 3 500 10 1,2,3,4,5,6,7,8 0,1 88 example_jaco_jan7
python3 ./src/potential_basis.py kinova 0,1 50 3 500 10 1,2,3,4,5,6,7,8 0,1 88 example_kinova_jan7
