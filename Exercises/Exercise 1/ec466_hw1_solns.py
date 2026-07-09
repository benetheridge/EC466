import numpy as np
import pandas as pd
import os
import sys

# Read the Excel file
# You need to have attend.xls in your working directory
os.chdir('C:/Users/bsethe/Dropbox/Work_non_research/Teaching/EC466/2025_26/Exercises/Exercise 1/')
# Best is to resave as attend.xlsx (Modern version)
A = pd.read_excel('attend.xlsx', header=None).values

# Question 1: Check the dimension of matrix A
print("Question 1 - Dimension of A:")
print(A.shape)
print()


# Question 2: Form y vector (final exam score) and X matrix (other variables)
# Assuming final exam score is in the first column (adjust index if different)
# y vector should include the dependent variable (final exam score)
# X matrix should include the rest of the variables

# HINT1: How to choose a vector from matrix A? 
# For instance, to choose termGPA variable from matrix A (second column):
# termGPA = A[:, 1]  (Python uses 0-based indexing)

# HINT2: How to merge two different vectors in one matrix
# If you have column vectors x1 and x2, then: X = np.column_stack([x1, x2])

# Example: If final score is first column and other variables are columns 2 onwards
y = A[:, 0]  # Adjust index based on your data structure
X = A[:, 1:]  # All other columns

print("Question 2 - y and X formed")
print(f"y shape: {y.shape}")
print(f"X shape: {X.shape}")
print()


# Question 3: Check the rank of X
rank_X = np.linalg.matrix_rank(X)
K = X.shape[1]
print("Question 3 - Rank of X:")
print(f"Rank of X: {rank_X}")
print(f"Number of columns (K): {K}")
print(f"Is rank equal to K? {rank_X == K}")
print()


# Question 4: Form matrix X'X (X transpose times X)
XtX = X.T @ X
print("Question 4 - X'X formed")
print(f"X'X shape: {XtX.shape}")
print()

sys.exit()


# Question 5: Take the inverse of X'X
print("Question 5 - Attempting to invert X'X:")
try:
    XtX_inv = np.linalg.inv(XtX)
    print("Inverse computed successfully")
except np.linalg.LinAlgError:
    print("ERROR: Matrix is singular and cannot be inverted")
    print("Reason: X'X is not full rank (has multicollinearity)")
print()

# Question 6: Remove regressors that are linear functions of other regressors
# You'll need to identify which columns are linearly dependent
# Example: If you determine columns need to be removed, create Xnew
# For demonstration, this is a placeholder - adjust based on your data
print("Question 6 - Creating Xnew (remove linearly dependent columns)")
print("You need to identify and remove linearly dependent columns")
# Example: Xnew = X[:, [0, 1, 3]]  # Keep columns 0, 1, 3, remove column 2
Xnew = X  # Placeholder - replace with actual column selection
print()

# Question 7: Check size and rank of Xnew
print("Question 7 - Size and rank of Xnew:")
print(f"Xnew shape: {Xnew.shape}")
rank_Xnew = np.linalg.matrix_rank(Xnew)
print(f"Rank of Xnew: {rank_Xnew}")
print(f"Is rank equal to number of columns? {rank_Xnew == Xnew.shape[1]}")
print()

# Question 8: Take the inverse of Xnew'Xnew
XnewTXnew = Xnew.T @ Xnew
XnewTXnew_inv = np.linalg.inv(XnewTXnew)
print("Question 8 - Inverse of Xnew'Xnew computed")
print()

# Question 9: Calculate OLS estimates
b = XnewTXnew_inv @ Xnew.T @ y
print("Question 9 - OLS estimates (b):")
print(b)
print()

# Question 10: Dimension of b
print("Question 10 - Dimension of b:")
print(f"b shape: {b.shape}")
print(f"Number of coefficients: {len(b)}")
print(f"Number of regressors: {Xnew.shape[1]}")
print(f"Same? {len(b) == Xnew.shape[1]}")
print()

# Question 11: Calculate predicted final scores
y_hat = Xnew @ b
print("Question 11 - Predicted values (y_hat):")
print(f"y_hat shape: {y_hat.shape}")
print()

# Question 12: Calculate residuals
e = y - y_hat
print("Question 12 - Residuals (e):")
print(f"Dimension of e: {e.shape}")
print()

# Question 13: (Same as Question 12)
print("Question 13 - Residuals (e):")
print(f"Dimension of e: {e.shape}")
print()

# Question 14: Average of residuals
mean_e = np.mean(e)
print("Question 14 - Average of residuals:")
print(f"Mean of e: {mean_e}")
print()

# Question 15: Average of y and y_hat
mean_y = np.mean(y)
mean_y_hat = np.mean(y_hat)
print("Question 15 - Averages:")
print(f"Average of y: {mean_y}")
print(f"Average of y_hat: {mean_y_hat}")
