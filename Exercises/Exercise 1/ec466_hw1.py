import numpy as np
import pandas as pd

# You need to have attend.xls in your working directory
A = pd.read_excel('attend.xls', header=None).values

## Question 1: To check the dimension of matrix A please write down A.shape


## Question 2: I would like you to estimate the final exam score on to the other variables. Please form a vector for y, and a matrix for X
# y vector should include the dependent variable (final exam score), X
# matrix should include the rest of the variables. Please read HINT1 and HINT2 first.

## HINT1: How to choose a vector from matrix A? For instance, I would like to choose termGPA variable from matrix A. Then I write that
# termGPA = A[:, 1] since termGPA is the second column of A (Python uses 0-based indexing, so column 2 is index 1).

## HINT2: How to merge two different vectors in one matrix. Suppose that I have column vectors x1 and x2, then I write that X = np.column_stack([x1, x2]).

## Question 3: After you form your matrix X (the nxK matrix of the regressors) please check the rank of X. Please type np.linalg.matrix_rank(X).
# Is it equal to K?

## Question 4: Next please form matrix X'X. You should type X.T @ X.

## Question 5: Please take the inverse of X'X. You should type np.linalg.inv(X.T @ X). Did Python give you an error? If yes, why?

## Question 6: Please remove the regressors, that are linear functions of the other regressors, from X. Define your new
# X matrix as Xnew.

## Question 7: Now please check the size and the rank of Xnew. Is the rank of Xnew equal to the number of its columns?

## Question 8: Please take the inverse of Xnew'Xnew.

## Question 9: Please calculate the OLS estimates for the model parameters. Hint: Please type b = np.linalg.inv(Xnew.T @ Xnew) @ Xnew.T @ y

## Question 10: What is the dimension of b? Hint: b.shape Is the number of coefficients the same as the number of regressors?

## Question 11: Please calculate the predicted final scores of students. Hint: the predicted final scores of students (y_hat) are y_hat = Xnew @ b

## Question 12: What are the residuals? What is the dimension of e? Hint: the residuals (e) are e = y - y_hat

## Question 13: What are the residuals? What is the dimension of e? Hint: the residuals (e) are e = y - y_hat.

## Question 14: What is the average of the residuals? Hint: type np.mean(e) to calculate the average

## Question 15: What is the average of y?


## What is the average of y_hat?
