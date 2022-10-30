from random import sample
import numpy as np
import pandas as pd 
from datetime import datetime
import time

#read data set
df = pd.read_csv('data_set.csv')
#convert date strings to datetime
df['date'] = pd.to_datetime(df['date'])
for i in range(len(df)):
    df['date'][i] = df['date'][i].date()

#create returns data frame
returns = {}

#create dataframe of arithmetic returns
for i in df.keys():
    if i != 'date':
        returns.update({str(i):df[i].pct_change()})
    else:
        returns.update({str(i):df[i]})
returns = pd.DataFrame.from_dict(returns)
#create sample mean vector
sample_mean_daily = {}
for i in returns.keys():
    if i != 'date':
        sample_mean_daily.update({i:[returns[i].mean()]})

sample_mean_daily = pd.DataFrame.from_dict(sample_mean_daily)

#using daily returns
gnk = returns['gnk'][1:].to_list()
bbby = returns['bbby'][1:].to_list()
hcci = returns['hcci'][1:].to_list()
skm = returns['skm'][1:].to_list()
cs = returns['cs'][1:].to_list()
azn = returns['azn'][1:].to_list()
nvda = returns['nvda'][1:].to_list()
epac = returns['epac'][1:].to_list()

X = np.stack((gnk,bbby,hcci,skm,cs,azn,nvda,epac), axis=0)
cov = np.cov(X)
ticker = ['gnk', 'bbby', 'hcci', 'skm', 'cs', 'azn', 'nvda', 'epac']
#cov_matrix = {'ticker':[], 'gnk':[], 'bbby':[], 'hcci':[], 'skm':[], 'cs':[], 'azn':[], 'nvda':[],'epac':[]}
cov_matrix = {'ticker':ticker}
for i in ticker:
    cov_matrix.update({i:[]})

for j in range(len(ticker)):
    for i in range(len(cov)):
        cov_matrix[ticker[j]].append(cov[i][j])

cov_matrix = pd.DataFrame.from_dict(cov_matrix)

print(sample_mean_daily.transpose())
print(cov_matrix)
# returns.to_csv('daily_arithmetic_returns.csv')
# sample_mean_daily.to_csv('sample_mean_daily_vector.csv')
# cov_matrix.to_csv('covariance_matrix_daily_returns.csv')

