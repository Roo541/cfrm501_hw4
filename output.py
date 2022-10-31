import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import row
import time

cov_matrix = pd.read_csv('annual_cov.csv')
mu_vector = pd.read_csv('mu_annual.csv')
mu_vector = mu_vector.drop(columns=['Unnamed: 0'])
#holder for weights once solved
asset_weights = mu_vector.keys()
asset_weights = asset_weights.to_list()
asset_weights = {'ticker':asset_weights[:], 'weight':[]}

#get sigma into array format
sigma = cov_matrix.copy()
sigma = sigma.set_index('ticker')
sigma = sigma.to_numpy()
#get return vector mu into array format
mu_vector_t = mu_vector.to_numpy()
mu_vector = np.transpose(mu_vector_t).reshape(8,1)
#calculate sigma_inverse
sigma_inverse = np.linalg.inv(sigma)

#define the unit vector and it's transpose
unit_vector_t = np.ones(8)
unit_vector = np.transpose(unit_vector_t).reshape(8,1)

#the equation to get weights
top_half = sigma_inverse.dot(unit_vector)
bottom_half = unit_vector_t.dot(sigma_inverse)
bottom_half = bottom_half.dot(unit_vector)

weights = top_half/bottom_half

for i in weights:
    asset_weights['weight'].append(i[0])
asset_weights = pd.DataFrame.from_dict(asset_weights)

#show that the weights add up to 1
summation = 0.0
for i in weights:
    summation += i[0]
print('summation of all weights: ', summation)
#output the weights
print('Weights w1 - w8: ')
print(asset_weights)

#transpose for future calculations
W_t = np.transpose(weights).reshape(1,8)
#calculate expected return
r_minima = W_t.dot(mu_vector)
print('expected return at minima: ' , r_minima[0][0])

# #calculate standard deviation, we know formula for Var = W^t Sigma W, we can calculate Var then find standard dev.
variance_minima = W_t.dot(sigma)
variance_minima = variance_minima.dot(weights)
print('Variance at expected minima: ', variance_minima[0][0])
print('Standard Deviation at expected minima: ', np.sqrt(variance_minima[0][0]))

#graphing the return sigma variance frontier
A = unit_vector_t.dot(sigma_inverse)
A = A.dot(unit_vector)

B = unit_vector_t.dot(sigma_inverse)
B = B.dot(mu_vector)

C = mu_vector_t.dot(sigma_inverse)
C = C.dot(mu_vector)

psi = A[0]*C[0] - B[0]**2
psi = psi[0]

mu_p = np.arange(-0.5,0.5,0.00001)
sigma_p = []

var_minima = variance_minima[0][0]
mu_minima = r_minima[0][0]

for i in mu_p:
    value = var_minima + ((i - mu_minima)**2)/(psi*var_minima)
    value = np.sqrt(value)
    sigma_p.append(value)

p1 = figure(plot_height = 600, plot_width = 800, 
        title = 'Mean Variance Frontier Curve',
        x_axis_label = 'sigma_p', 
        y_axis_label = 'mu_p')
p2 = figure(plot_height = 600, plot_width = 800, 
        title = 'Mean Variance Frontier Curve',
        x_axis_label = 'sigma_p', 
        y_axis_label = 'mu_p')

p1.line(sigma_p, mu_p, line_width=2, color='blue', legend_label='r-sigma graph')
p2.line(sigma_p, mu_p, line_width=2, color='blue', legend_label='r-sigma graph')

#show(row(p1, p2))

#for investor desired return of 0.25 calculate the std dev and weights
investor_std_dev = np.sqrt(var_minima + ((0.25 - mu_minima)**2)/(psi*var_minima))
print('std dev for expected portfolio return of 0.25:', investor_std_dev)

#calculate weights
lambda_1 = (B[0]*0.25 - C[0])/psi
lambda_2 = (B[0]-A[0]*0.25)/psi

part_1_w = sigma_inverse.dot(unit_vector)
part_1_w = part_1_w*(-lambda_1)

part_2_w = sigma_inverse.dot(mu_vector)
part_2_w = part_2_w*(-lambda_2)

weights_solution = part_1_w + part_2_w
solution_weights = {'ticker':asset_weights['ticker'][:], 'weight':[]}
for i in weights_solution:
    solution_weights['weight'].append(i[0])

solution_weights = pd.DataFrame.from_dict(solution_weights)
print('solution of weights for return of 0.25:')
print(solution_weights)
summation = 0.00
for i in solution_weights['weight']:
    summation += i
print('summation of weights:', summation)

#now investor wants to achieve std dev of 0.23, what is the best return available at this value?
print('psi is:', psi)

