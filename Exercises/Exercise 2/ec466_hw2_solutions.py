import numpy as np
import pandas as pd
import sys

print('Your Name')
print('Exercise 2, Question 1')
print()

# You need to have beauty.xls in your working directory
A = pd.read_excel('M:/Teaching/EC466/Class2/beauty.xlsx', header=None).values

## Question 1: What is the number of observations? How many variables are there in A?
print("Question 1:")
print(f"Dimension of A: {A.shape}")
print(f"Number of observations (rows): {A.shape[0]}")
print(f"Number of variables (columns): {A.shape[1]}")
print()


## Question 2: I would like you to estimate the log(wages) on to the other variables. 
# Please form a vector for y, and a matrix for X
# y vector should include the dependent variable (log(wages)), X
# matrix should include the rest of the variables. This time please add an
# intercept to your model.

# Assuming log(wages) is in the first column (adjust if different)
y = A[:, 0]

# Create X matrix with intercept as first column, then other variables
intercept = np.ones((len(y), 1))
X = np.column_stack([intercept, A[:, 1:]])

print("Question 2:")
print(f"y shape: {y.shape}")
print(f"X shape: {X.shape}")
print()

## Question 3: After you form your matrix X (the nxK matrix of the regressors) 
# please check the rank of X.
rank_X = np.linalg.matrix_rank(X)
K = X.shape[1]

print("Question 3:")
print(f"Rank of X: {rank_X}")
print(f"Number of columns K: {K}")
print(f"Is rank equal to K? {rank_X == K}")
print()


## Question 4: Estimate the model by the OLS estimator.
b = np.linalg.inv(X.T @ X) @ X.T @ y

print("Question 4:")
print("OLS estimates (b):")
print(b)
print()

# Also do using "canned" tools
import statsmodels.api as sm
model = sm.OLS(y, X)
results = model.fit()
print("Model Summary:")
print(results.summary())


## Question 5: Compute the projection matrix, P.
P = X @ np.linalg.inv(X.T @ X) @ X.T

print("Question 5:")
print(f"Projection matrix P computed")
print(f"P shape: {P.shape}")
print()


## Question 6: Compute the annihilator matrix M.
M = np.eye(len(P)) - P

print("Question 6:")
print(f"Annihilator matrix M computed")
print(f"M shape: {M.shape}")
print()


## Question 7: Compute the inverse of P and M. What are your results?
print("Question 7:")
print("Attempting to compute inverse of P:")
try:
    P_inv = np.linalg.inv(P)
    print("P inverse computed (but this shouldn't work for non-full rank matrix)")
except np.linalg.LinAlgError:
    print("ERROR: P is singular and cannot be inverted")
    print("Reason: P is a projection matrix and not full rank (unless X is square and full rank)")

print("\nAttempting to compute inverse of M:")
try:
    M_inv = np.linalg.inv(M)
    print("M inverse computed (but this shouldn't work for non-full rank matrix)")
except np.linalg.LinAlgError:
    print("ERROR: M is singular and cannot be inverted")
    print("Reason: M is also a projection matrix and not full rank")
print()

print("Condition number of M")
print(np.linalg.cond(M_inv))
print()


## Question 8: Compute the determinant of M and P.
det_P = np.linalg.det(P)
det_M = np.linalg.det(M)

print("Question 8:")
print(f"Determinant of P: {det_P}")
print(f"Determinant of M: {det_M}")
print("Note: Both determinants should be 0 (or very close to 0) because P and M are singular projection matrices")
print()


## Question 9: Compute the predicted log(wages) by defining yhat as the predicted values, 
# then compare your solution with P@y.
yhat = X @ b
# Can also use Statsmodel
yhat_2 = results.fittedvalues
yhat_3 = results.predict(X)
comparison = np.column_stack([yhat, yhat_2, yhat_3, P @ y, yhat - P @ y])

print("Question 9:")
print("Comparison of yhat, P@y, and their difference (first 10 rows):")
np.set_printoptions(precision=4, suppress=True)
print(comparison[:10])
print(f"Maximum difference between yhat and P@y: {np.max(np.abs(yhat - P @ y))}")
print("Note: yhat and P@y should be identical (difference should be essentially 0)")
print()

## Question 10: Compute the residuals for the model by defining e as the residuals, 
# then compare your solution with M@y.
e = y - yhat
comparison_resid = np.column_stack([e, M @ y, e - M @ y])

print("Question 10:")
print("Comparison of e, M@y, and their difference (first 10 rows):")
print(comparison_resid[:10])
print(f"Maximum difference between e and M@y: {np.max(np.abs(e - M @ y))}")
print("Note: e and M@y should be identical (difference should be essentially 0)")
print()


## Question 11: Now we will calculate the estimate for Var(b|X) (call it Varbhat). 
# We know that Varbhat=s2*inv(X'*X) where s2=e'*e/(n-rank(X))
n = len(y)
s2 = (e.T @ e) / (n - rank_X)
Varbhat = s2 * np.linalg.inv(X.T @ X)

print("Question 11:")
print(f"n = {n}")
print(f"rank(X) = {rank_X}")
print(f"s2 (variance estimate) = {s2}")
print(f"Varbhat computed")
print()


## Question 12: What is the dimension of Varbhat?
print("Question 12:")
print(f"Dimension of Varbhat: {Varbhat.shape}")
print(f"Varbhat is a {Varbhat.shape[0]}x{Varbhat.shape[1]} matrix")
print()


## Question 13: Next, we compute the standard errors of the coefficients. 
# Take the square root of each element on Varbhat (element-wise square root). 
# Then extract the diagonal elements.
sqVbhat = Varbhat**(1/2)
se = np.diag(sqVbhat)

print("Question 13:")
print("Standard errors of coefficients:")
print(se)
print()


# Display results with b and se
result = np.column_stack([b, se])
print("Results [coefficients, standard errors]:")
print(result)
print()


## Question 14: Now compute the t-values for the coefficients for the null hypothesis that b_i=0.
tval = b / se

print("Question 14:")
print("t-values for each coefficient:")
print(tval)
print()

# Display full results
result_full = np.column_stack([b, se, tval])
print("Full Results [coefficients, standard errors, t-values]:")
print(result_full)
print()

## Write down which variables are significant or not.
print("Statistical Significance (using |t| > 1.96 for 5% significance level):")
for i in range(len(tval)):
    significant = "SIGNIFICANT" if np.abs(tval[i]) > 1.96 else "NOT significant"
    if i == 0:
        print(f"Variable {i} (Intercept): t = {tval[i]:.4f} - {significant}")
    else:
        print(f"Variable {i}: t = {tval[i]:.4f} - {significant}")


