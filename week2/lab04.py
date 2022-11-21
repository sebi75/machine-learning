import numpy as np
import matplotlib.pyplot as plt
from lab03 import zscore_normalize_features
from main import run_gradient_descent_feng

# this lab is dedicated to feature engineering

np.set_printoptions(precision=2)  # reduces the precision fon numpy arrays

x = np.arange(0, 20, 1) # initiate an array from 0 to 20 - 1 elements going with step 1
print(f"aranged")
print(x)

y = 1 + x ** 2

x_train = x.reshape(-1, 1) # takes all the elements in x and makes as mant rows with 1 column
print(f"x_train: {x_train}")


