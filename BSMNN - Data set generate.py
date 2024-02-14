import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import keras
import keras.backend as K


from sklearn.model_selection import ParameterGrid
from py_vollib import black_scholes_merton as bsm
from progressbar import ProgressBar
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import uniform
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
# S (spot price)
def thisS (q):
 return gamma.ppf(q, a = 100, scale = 1)
# K (strike price)
# uniform (lower = 50, upper = 200)
def thisK (q):
 return uniform.ppf(q, 50, 200)
# (interest rate)
# uniform (lower = 0.01, upper = 0.18)
def thisR (q):
 return uniform.ppf(q, 0.01, 0.18)
# D (dividend)
# uniform (lower = 0.01, upper = 0.18)
def thisD (q):
 return uniform.ppf(q, 0.01, 0.18)
17
# t (time-to-maturity)
# t will be 3, 6, 9, 12 months for all examples (0.25, 0.5, 0.75, 1 year)
# sigma (volatility)
# beta (add small amount so volatility cannot be zero)
def thisSigma (q):
 return (beta.ppf(q, a = 2, b = 5) + 0.001)
num_increment = 12
percentiles = pd.Series(np.linspace(0, 0.99, num_increment))
S = percentiles.apply(thisS).tolist()
K = percentiles.apply(thisK).tolist()
q = percentiles.apply(thisD).tolist()
t = np.array([.25, .5, .75, 1])
r = percentiles.apply(thisR).tolist()

S.pop(0)
sigma = percentiles.apply(thisSigma).tolist()
param_grid = {'S': S, 'K' : K, 'q' : q, 't' : t, 'r' : r, 'sigma' : sigma}
grid = ParameterGrid(param_grid)


pbar = ProgressBar()
fullDF = pd.DataFrame(columns =['K','S','q','r','sigma','t','price'])
prices = []

for params in pbar(grid):
 price = bsm.black_scholes_merton(flag='p', S=params['S'], K=params['K'], q=params['q'], t=params['t'], r=params['r'], sigma=params['sigma'])
 price = round(price,2)
 if price < 0.01:
  price = 0.0
  params['price'] = price
 else:
  params['price'] = price
 temp_df = pd.DataFrame([params])
 fullDF = pd.concat([fullDF, temp_df], ignore_index=True)




# output to csv
fullDF.to_csv('C:\\Users\\yezda\\OneDrive\\Python Virtual Environments\\BlackScholeML\\BSMNNvenv\\dataFullV2.csv', index = False)
print('csv has been saved to venv file!')
#Warning to anyone who would like to replicate this, the data set will be quite large, might be worth it to migrate it to cloud storage to save local storage space.
