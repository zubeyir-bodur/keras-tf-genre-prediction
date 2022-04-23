"""
Contributors only - Please check if your IDE can run this script
If there are no errors, this will indicate that all packages were installed
Otherwise, there is a problem.
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas
import keras
import seaborn

# Defining Array 1
a = np.array([[1, 2],
              [3, 4]])

# Defining Array 2
b = np.array([[4, 3],
              [2, 1]])

# Adding 1 to every element
print("Adding 1 to every element:", a + 1)

# Subtracting 2 from each element
print("\nSubtracting 2 from each element:", b - 2)

# sum of array elements
# Performing Unary operations
print("\nSum of all array "
      "elements: ", a.sum())

# Adding two arrays
# Performing Binary operations
print("\nArray sum:\n", a + b)

# Implementation of matplotlib function

# First create some toy data:
x = np.linspace(0, 1.5 * np.pi, 100)
y = np.sin(x ** 2) + np.cos(x ** 2)

plt.figure(figsize=(16, 9), dpi=300)
fig, axs = plt.subplots(2, 2, subplot_kw=dict(polar=True))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

fig.suptitle('matplotlib.pyplot.subplots() Example')
plt.show()
