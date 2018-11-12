# -*- coding: utf-8 -*-
"""
MIE1613 Project by Xiaotian Zhu 1003250545
"""

# Pricing American put option using Monte Carlo simulation
import math
import RNG
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

S0 = 100      # initial price of underlying stock 
K = 110       # strike price
sigma = 0.34641  # volatility
r = 0.1      # risk-free rate
T = 1/3      # maturity in years

#Generate stock price sample paths using Monte Carlo
m = 13      # number of time periods
n1 = 1000   # number of in-sample paths
n2 = 5000   # number of out-of-sample paths
dT = T/m

# Computes numPaths random paths for a GBM
# T is the total length of time for the path (in years)
# dT is the time increment (in years)

# Vector of paths will store realizations of the asset price
# First asset price is the initial price S0
paths = np.zeros((n1,m+1))
paths[:,0] = S0

# Generate n1 in-sample paths using GBM
for iPath in range (n1):
    for iStep in range(m):
        factor1 = math.exp((r-0.5*sigma**2)*dT+sigma*math.sqrt(dT)*RNG.Normal(0,1,1)) 
        paths[iPath,iStep+1]=paths[iPath,iStep]*factor1

# Generate n2 out-of-sample paths using GBM
S = np.zeros((n2,m+1))
S[:,0] = S0
for iPath in range(n2):
    for iStep in range(m):
        factor2 = math.exp((r-0.5*sigma**2)*dT+sigma*math.sqrt(dT)*RNG.Normal(0,1,10))
        S[iPath,iStep+1]=S[iPath,iStep]*factor2

# Pricing American put option using Monte Carlo Simulation
theta = np.zeros(m+1)  # exercise boudary at time t_j
theta_estimate = np.zeros(m+1)
Z = np.zeros((n1,m+1))  # payoff matrix
P = np.zeros(m+1)
P_estimate = np.zeros(m+1)

theta[m] = K
theta_estimate[m] = K
Z[:,m] = np.maximum(K-paths[:,m].T,np.zeros((n1))).T
P[m] = np.mean(Z[:,m])
P_estimate[m] = np.mean(Z[:,m])

# (3): obtain the exercise boundary
for j in range(m-1,-1,-1):
    #(a):
    theta[j] = theta[j+1]
    theta_estimate[j] = theta[j]
    
    #(b) compute P_estimate:
    for ii in range(n1):
        # find the optimal stopping time
        tau_j = j
        while tau_j < m:
            if paths[ii,tau_j] <= theta[j]:
                break
            else:
                tau_j += 1
        Z[ii,j] = math.exp(-r*(tau_j-j)*dT) * max(K-paths[ii,tau_j],0)
    P_estimate[j] = np.sum(Z[:,j])/n1
    
    #(c): update boundary estimations
    for i in range(n1):
        if paths[i,j] < theta[j+1]:
            theta[j] = paths[i,j]
            for ii in range(n1):
                # find the optimal stopping time
                tau_j = j
                while tau_j < m:
                    if paths[ii,tau_j] <= theta[j]:
                        break
                    else:
                        tau_j += 1
                Z[ii,j] = math.exp(-r*(tau_j-j)*dT) * max(K-paths[ii,tau_j],0)
            P[j] = np.sum(Z[:,j])/n1
            if P[j] > P_estimate[j]:
                theta_estimate[j] = theta[j]
                P_estimate[j] = P[j]
   
    #(d):
    theta[j] = theta_estimate[j] 
'''
# Golden-section Search for fingding optimal theta:
gr = (math.sqrt(5) + 1) / 2 # golden section point
tol = 2
for j in range(m-1,-1,-1):
    #(a):
    theta[j] = theta[j+1]
    theta_estimate[j] = theta[j]

    a = 0
    b = theta[j]
    c = b - (b - a) / gr
    d = a + (b - a) / gr 
    while abs(c - d) > tol:
        # find f(c):
        for ii in range(n1):
            # find the optimal stopping time
            tau_j = j
            while tau_j < m:
                if paths[ii,tau_j] <= c:
                    break
                else:
                    tau_j += 1
                Z[ii,j] = math.exp(-r*(tau_j-j)*dT) * max(K-paths[ii,tau_j],0)
            fc = np.sum(Z[:,j])/n1
        
        # find f(d):
        for ii in range(n1):
            # find the optimal stopping time
            tau_j = j
            while tau_j < m:
                if paths[ii,tau_j] <= d:
                    break
                else:
                    tau_j += 1
                Z[ii,j] = math.exp(-r*(tau_j-j)*dT) * max(K-paths[ii,tau_j],0)
            fd = np.sum(Z[:,j])/n1            
        
        if fc < fd:
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    theta[j]  = (b + a) / 2    
'''

#(4): pricing American put option: in-sample
tau_in = np.zeros(n1)
h_in = np.zeros(n1)
for i in range(n1):
    #(a):
    j = 1
    
    #(b):
    while j <= m:
        #(i):
        if paths[i,j] <= theta[j]:
            tau_in[i] = j
            break
        #(ii):
        else:
            j += 1

    #(c):
    if tau_in[i] == 0:
        tau_in[i] = m
    
    #(d):
    h_in[i] = math.exp(-r*(tau_in[i])*dT) * max(K-paths[i,int(tau_in[i])],0) 

#(4): pricing American put option: out-of-sample
tau_out = np.zeros(n2)
h_out = np.zeros(n2)
for i in range(n2):
    #(a):
    j = 1
    
    #(b):
    while j <= m:
        #(i):
        if S[i,j] <= theta[j]:
            tau_out[i] = j
            break
        #(ii):
        else:
            j += 1

    #(c):
    if tau_out[i] == 0:
        tau_out[i] = m
    
    #(d):
    h_out[i] = math.exp(-r*(tau_out[i])*dT) * max(K-S[i,int(tau_out[i])],0) 


#(5): optimal stopping time
tau_in_bar = np.mean(tau_in)
tau_in_var = np.var(tau_in)
tau_out_bar = np.mean(tau_out)
tau_out_var = np.var(tau_out)

#(6): option price
h_in_bar = np.mean(h_in)
h_in_var =np.var(h_in)
h_out_bar = np.mean(h_out)
h_out_var =np.var(h_out)

aux = range(1,len(h_out)+1)
h_cumavg = np.cumsum(h_out)/aux
plt.plot(aux,h_cumavg)
plt.xlabel('# replications')
plt.ylabel('Running average option values')
plt.show()

print('95% CI, in-sample: (',h_in_bar-1.96*math.sqrt(h_in_var/n2),',',h_in_bar+1.96*math.sqrt(h_in_var/n2),')')
print('95% CI, out-of-sample: (',h_out_bar-1.96*math.sqrt(h_out_var/n2),',',h_out_bar+1.96*math.sqrt(h_out_var/n2),')')

plt.hist(h_out,bins=30)
plt.xlabel('Option Value')
plt.ylabel('# Observations')
plt.show()
plt.hist(tau_out,bins=m)
plt.xlabel('Optimal tau')
plt.ylabel('# Observations')
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))



# Plot sample paths and the exercise boundary:
plt.plot(S.T)
plt.plot(theta.T,'black')
plt.xlabel('Time Period')
plt.ylabel('Stock Price/$')
plt.xlim(0,m) 
plt.title('Stock Price Sample Paths')
plt.show()








