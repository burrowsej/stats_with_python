import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# covariance
dataset = pd.read_csv("height_weight.csv", usecols=['height', 'weight'])
dataset.head()

covariance = np.cov(dataset.T)# need to transpose or use rowvar=False
print(covariance)

covariance = dataset.cov()
print(covariance) # much cleaner than numpy

# correlation
# divide the cov matrix by the product of the sds
# -1 up to 1
correlation = np.corrcoef(dataset.T)
correlation = dataset.corr()
print(correlation)

# not many other ways of summarising nd relationships down to a few numbers
# this can only deal with very simple relationships
# does not mean we can not exploit complex relationships in hypothesis testing
