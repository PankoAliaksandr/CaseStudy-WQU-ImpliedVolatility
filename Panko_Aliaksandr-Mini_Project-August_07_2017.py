
# coding: utf-8

# # Mini Project: Calculate the implied volatility using Newton Raphson Algorithm
# 
# ## Implied volatility calculation formula:
# **Implied volatility** is calculated by taking the market price of the option, entering it into the B-S formula, and back-solving for the value of the volatility.
# **Model Inputs**
# * the market price of the option.
# * the underlying stock price.
# * the strike price.
# * the time to expiration.
# * the risk-free interest rate.
# 
# [BS model and implementation](http://www.espenhaug.com/black_scholes.html)
# 
# ## Newton's method
# $$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $$
# [Method description](https://en.wikipedia.org/wiki/Newton%27s_method)

# In[31]:


# Libraries
import numpy as np
from scipy.stats import norm

# Initialization
S0 = 34    # Stock price
E = 34     # Strike priece
r = 0.001  # Risk-free
t = 1      # Time to expiration
c = 2.7240 # Call option price

# Start of calculations. Initial guess:
sigma = 0.10
i = 0

# Newton-Raphson method
while("true"):
    # D1 in BS formula:
    d1 = (np.log(S0  /E) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
    # D2 in BS formula:
    d2 = d1 - sigma * np.sqrt(t)
    # Call option price function with LHS = 0 (required form for Newton algorithm)
    f = S0 * norm.cdf(d1) - E * np.exp(-r * t) * norm.cdf(d2) - c

    # Derivative of d1 w.r.t. sigma:
    d11 = (sigma ** 2 * t * np.sqrt(t) - (np.log(S0 / E) + (r + sigma ** 2 / 2) * t) * np.sqrt(t)) / (sigma ** 2 * t)
    # Derivative of d2 w.r.t. sigma:
    d22 = d11 - np.sqrt(t)
    # Derivative of f(sigma):
    f1 = S0 * norm.cdf(d1) * d11 - E * np.exp(-r * t) * norm.cdf(d2) * d22
    # Update sigma:
    sigma_new = sigma - f / f1

    i = i+1

    # Stop after 100 iterations anyway
    if (i == 100 or abs(sigma_new - sigma) < 0.00000001):
        print("implied volatility is:", sigma_new)   
        break
    else:
        sigma = sigma_new

