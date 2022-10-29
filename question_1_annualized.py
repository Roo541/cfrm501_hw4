import numpy as np
import pandas as pd

#get the covariance matrix daily arithmetic returns not pricing
cov = pd.read_csv('covariance_matrix_daily_returns.csv')
cov = cov.drop(columns = 'Unnamed: 0')
cov = cov.set_index('ticker')

#get the sample returns for each stock df
mean_vector = pd.read_csv('sample_mean_daily_vector.csv')
mean_vector = mean_vector.drop(columns = 'Unnamed: 0')

annual_cov = cov.copy()

#Go through each element in the covariance matrix and apply the formula for annualizing
N = 252
tickers = cov.keys()
for j in cov.keys():
    mu_j = mean_vector[j][0]    # the columns
    for i in tickers:
        mu_i = mean_vector[i][0]    #the rows
        cov_ij = cov[j][i]
        value = ((cov_ij + (mu_i + 1)*(mu_j + 1))**N) -((mu_i + 1)**N)*((mu_j + 1)**N)
        annual_cov[j][i] = value

print(cov)
print(annual_cov)