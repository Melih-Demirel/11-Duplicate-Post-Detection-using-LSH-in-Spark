'''
    When we want a specific threshold, we need to find 2 variables in order to calculate our treshold:
        - Number of signatures in MinHash
        - Number of BANDS
    The formula for calculating the treshold: (1 / BANDS) ** (1 / (NUMBER OF SIGNATURES // BANDS))
    This script finds values for both variables that are ranged between the given threshold: [0.59, 0.61] in this case.
'''

import numpy as np
import matplotlib.pyplot as plt




desired_range = (0.59, 0.61)



def equation(x, y):
    return (1 / x) ** (1 / (y / x))

x_values = np.linspace(1, 100, 100)
y_values = np.linspace(1, 300, 300)

x_mesh, y_mesh = np.meshgrid(x_values, y_values)

z_values = equation(x_mesh, y_mesh)



indices = np.where((z_values >= desired_range[0]) & (z_values <= desired_range[1]))
x_within_range = x_mesh[indices]
y_within_range = y_mesh[indices]

print(f"For equation values between {desired_range[0]:.2f} and {desired_range[1]:.2f}:")
for x_val, y_val in zip(x_within_range, y_within_range):
    if y_val % x_val == 0:
        print(f"Size of signatures MinHash = {y_val:.2f}, Bands = {x_val:.2f}, Threshold = {(1 / x_val) ** (1 / (y_val / x_val))}")