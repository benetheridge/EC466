import numpy as np
import pandas as pd

print('your name')
print('Exercise 2, Question 1')

# You need to have beauty.xls in your working directory
A = pd.read_excel('beauty.xls', header=None).values

## Question 1: What is the number of observations? How many variables are there in A?



## Question 2: I would like you to estimate the log(wages) on to the other variables. Please form a vector for y, and a matrix for X
# y vector should include the dependent variable (log(wages)), X
# matrix should include the rest of the variables. This time please add an
# intercept to your model. Hint: Add a vector of ones in the first column
# of X. To obtain a vector of ones that is 1260x1 type np.ones((len(y), 1)).




## Question 3: After you form your matrix X (the nxK matrix of the regressors) please check the rank of X. Please type np.linalg.matrix_rank(X).
# Is it equal to K?



## Question 4: Estimate the model by the OLS estimator.



## Question 5: Compute the projection matrix, P. Type P = X @ np.linalg.inv(X.T @ X) @ X.T


## Question 6: Compute the annihilator matrix M. Type M = np.eye(len(P)) - P



## Question 7: Compute the inverse of P and M. What are your results?



## Question 8: Compute the determinant of M and P. Type np.linalg.det(P) and np.linalg.det(M)



## Question 9: Compute the predicted log(wages) by defining yhat as the predicted values, then compare your solution with P@y. Hint: Type np.column_stack([yhat, P@y, yhat-P@y])



## Question 10: Compute the residuals for the model by defining e as the residuals, then compare your solution with M@y. Hint: Type np.column_stack([e, M@y, e-M@y])



## Question 11: Now we will calculate the estimate for Var(b|X) (call it Varbhat). We know that Varbhat=s2*inv(X'*X) where s2=e'*e/(n-rank(X))

#n = len(y)


## Question 12: What is the dimension of Varbhat? 



## Question 13: Next, we compute the standard errors of the coefficients. Take the square root of each element on Varbhat (THIS IS DIFFERENT THAN TAKING THE SQUARE ROOT OF A MATRIX). Then extract the diagonal elements.
# type sqVbhat = Varbhat**(1/2) and se = np.diag(sqVbhat)



## The result you find above is called the standard errors of b. See the below command
#result = np.column_stack([b, se])

## Question 14: Now compute the t-values for the coefficients for the null hypothesis that b_i=0. tval = b/se


#result = np.column_stack([b, se, tval])

## Write down which variables are significant or not.
