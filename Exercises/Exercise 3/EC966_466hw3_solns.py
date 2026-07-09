
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
import os


print('Exercise 3, Question 1')

# You need to copy and paste beauty.xls file to the folder in your computer.
A = pd.read_excel('M:/Teaching/EC466/Class3/hprice.xlsx', header=None).values

# Question 1: What is the number of observations? How many variables are there in A?

sizeA = A.shape

# Question 2: I would like you to estimate the log(prices) on log(assess),
# log(lotsize), log(sqrft), bdrms, and colonial. Please form a vector for y, and a matrix for X
# X matrix should include a constant and the regressors. Hint: Add a vector of ones in the first column
# of X.

y = A[:, 6]
X = np.column_stack([np.ones(len(y)), A[:, 7:10], A[:, 2], A[:, 5]])


# Question 3: After you form your matrix X (the nxK matrix of the regressors) please check the rank of X. Please type rank(X).
# Is it equal to K?
K = np.linalg.matrix_rank(X)
print(f"X has {X.shape[1]} columns")
print(f"Rank of X is equal to: {K}")


# Question 4: Estimate the model by the OLS estimator.

## 4.1  Using matrix formula
b = np.linalg.inv(X.T @ X) @ X.T @ y

## 4.2  Using stats models - basic
model = sm.OLS(y, X)
results = model.fit()
print("Model Summary:")
print(results.summary())

## 4.3 Using stats models - converting X to a dataframe
# Convert X to a DataFrame with column names
Xv2 = pd.DataFrame(X, columns=['Cons', 'ln Assess', 'ln Lotsize', 'ln Sqrft', 'bdrms', 'colonial'])  # adjust names and count as needed
modelv2 = sm.OLS(y, Xv2)
results = modelv2.fit()
print("Model Summary:")
print(results.summary())


# Question 5: Compute the standard errors for the coefficients

n = len(y)
e = y - X @ b
s2 = e.T @ e / (n - K)
Varbhat = s2 * np.linalg.inv(X.T @ X)
Varbhat_diag = np.diag(Varbhat)
se = np.sqrt(Varbhat_diag)
print(f"\nStandard errors (should be same as regression printout from Stats Models):\n {se}")


# Question 6: Compute the t-values for each coefficients

tvalues = b / se
print(f"\nT-values:\n {tvalues}")


# Question 7: Test the null hypothesis of insignificance for each regressors at 5% level of significance.

# Ans: llotsize is the only variable that is significant.


# Question 8: Now test the null hypothesis for the joint significance. First, calculate the SSR for the unrestricted regression
# then run the regression on the restricted regression (Hint: you should come up with a new X, which only includes a vector of ones)
# Now you should also calculate the SSR for the restricted regression. Then
# type the formula for the F-stat to compute the Fstatistics.

ssrU = e.T @ e

Xnew = np.ones((n, 1))

br = np.linalg.inv(Xnew.T @ Xnew) @ Xnew.T @ y

er = y - Xnew @ br

ssrR = er.T @ er

Fstat = ((ssrR - ssrU) / (K - 1)) / (ssrU / (n - K))

print(f"\nFstat:\n {Fstat}")




# Question 9: Someone suggested that the coefficients of log(assess) and log(lotsize) should sum up to 1. Please test this hypothesis at 5% level of significance
# You should create a matrix R and a vector r as in the lecture notes.

R = np.array([[0, 1, 1, 0, 0, 0]])
r = 1

Fstat1 = (R @ b - r).T @ np.linalg.inv(R @ Varbhat @ R.T) @ (R @ b - r) / np.linalg.matrix_rank(R)

## Use scipy to compute CDF of F-distribution
from scipy import stats

print("\n9: The test statistic in the final test has a F(1,82) distribution under the null.")
print(f"\nFstat1:\n {Fstat1}")
print(f"\np-value: {1 - stats.f.cdf(Fstat1, dfn=1, dfd=82)}")


