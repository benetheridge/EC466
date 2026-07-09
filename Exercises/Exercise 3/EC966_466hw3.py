
import numpy as np
import pandas as pd
import statsmodels.api as sm


# You need to copy and paste beauty.xls file to the folder in your computer.
A = pd.read_excel('hprice.xlsx', header=None).values

# Question 1: What is the number of observations? How many variables are there in A?


# Question 2: I would like you to estimate the log(prices) on log(assess),
# log(lotsize), log(sqrft), bdrms, and colonial. Please form a vector for y, and a matrix for X
# X matrix should include a constant and the regressors. Hint: Add a vector of ones in the first column
# of X.


# Question 3: After you form your matrix X (the nxK matrix of the regressors) please check the rank of X. Please type rank(X).
# Is it equal to K?


# Question 4: Estimate the model by the OLS estimator.


# Question 5: Compute the standard errors for the coefficients


# Question 6: Compute the t-values for each coefficients


# Question 7: Test the null hypothesis of insignificance for each regressors at 5% level of significance.


# Question 8: Now test the null hypothesis for the joint significance. First, calculate the SSR for the unrestricted regression
# then run the regression on the restricted regression (Hint: you should come up with a new X, which only includes a vector of ones)
# Now you should also calculate the SSR for the restricted regression. Then
# type the formula for the F-stat to compute the Fstatistics.


# Question 9: Someone suggested that the coefficients of log(assess) and log(lotsize) should sum up to 1. Please test this hypothesis at 5% level of significance
# You should create a matrix R and a vector r as in the lecture notes.



