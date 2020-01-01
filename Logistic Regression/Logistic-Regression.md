Logistic Regression
================
David Kwon

# Logistic Regression

## Binomial Distribution

Let ![Y](https://latex.codecogs.com/png.latex?Y "Y") = number of
successes in ![m](https://latex.codecogs.com/png.latex?m "m") trials of
a binomial process. Then Y is said to have a binomial distribution with
parameters ![m](https://latex.codecogs.com/png.latex?m "m") and
![\\theta](https://latex.codecogs.com/png.latex?%5Ctheta "\\theta").

  
![Y \\sim Bin(m,
\\theta)](https://latex.codecogs.com/png.latex?Y%20%5Csim%20Bin%28m%2C%20%5Ctheta%29
"Y \\sim Bin(m, \\theta)")  
The probability that ![Y](https://latex.codecogs.com/png.latex?Y "Y")
takes the integer value ![j](https://latex.codecogs.com/png.latex?j "j")
(![j = 0, 1, ...,
m](https://latex.codecogs.com/png.latex?j%20%3D%200%2C%201%2C%20...%2C%20m
"j = 0, 1, ..., m")) is given by

  
![P(Y=j) = \\binom{m}{j} \\theta^j
(1-\\theta)^{m-j}](https://latex.codecogs.com/png.latex?P%28Y%3Dj%29%20%3D%20%5Cbinom%7Bm%7D%7Bj%7D%20%5Ctheta%5Ej%20%281-%5Ctheta%29%5E%7Bm-j%7D
"P(Y=j) = \\binom{m}{j} \\theta^j (1-\\theta)^{m-j}")  
Then,

let ![y\_i](https://latex.codecogs.com/png.latex?y_i "y_i") = number of
successes in ![m\_i](https://latex.codecogs.com/png.latex?m_i "m_i")
trials of a binomial process where ![i
= 1,...,n](https://latex.codecogs.com/png.latex?i%20%3D%201%2C...%2Cn
"i = 1,...,n").

Then,

  
![(Y = y\_i|X = x\_i) \\sim Bin(m\_i,
\\theta(x\_i))](https://latex.codecogs.com/png.latex?%28Y%20%3D%20y_i%7CX%20%3D%20x_i%29%20%5Csim%20Bin%28m_i%2C%20%5Ctheta%28x_i%29%29
"(Y = y_i|X = x_i) \\sim Bin(m_i, \\theta(x_i))")  

Sigmoid function is chosen to differentiate the cost function.

## Sigmoid function and Log Odds

  
![\\theta(x) = \\frac{e^{\\beta\_0 + \\beta\_1x}}{1 + e^{\\beta\_0 +
\\beta\_1x}} = \\frac{1}{1 + e^{-(\\beta\_0 +
\\beta\_1x)}}](https://latex.codecogs.com/png.latex?%5Ctheta%28x%29%20%3D%20%5Cfrac%7Be%5E%7B%5Cbeta_0%20%2B%20%5Cbeta_1x%7D%7D%7B1%20%2B%20e%5E%7B%5Cbeta_0%20%2B%20%5Cbeta_1x%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20e%5E%7B-%28%5Cbeta_0%20%2B%20%5Cbeta_1x%29%7D%7D
"\\theta(x) = \\frac{e^{\\beta_0 + \\beta_1x}}{1 + e^{\\beta_0 + \\beta_1x}} = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1x)}}")  
Here, the
![\\theta(x)](https://latex.codecogs.com/png.latex?%5Ctheta%28x%29
"\\theta(x)") is the probability of success. Solving the equation for
![\\beta\_0 +
\\beta\_1x](https://latex.codecogs.com/png.latex?%5Cbeta_0%20%2B%20%5Cbeta_1x
"\\beta_0 + \\beta_1x"),

  
![\\beta\_0 + \\beta\_1x = log(\\frac{\\theta(x)}{1-\\theta(x)})
](https://latex.codecogs.com/png.latex?%5Cbeta_0%20%2B%20%5Cbeta_1x%20%3D%20log%28%5Cfrac%7B%5Ctheta%28x%29%7D%7B1-%5Ctheta%28x%29%7D%29%20
"\\beta_0 + \\beta_1x = log(\\frac{\\theta(x)}{1-\\theta(x)}) ")  
Here, the
![log(\\frac{\\theta(x)}{1-\\theta(x)})](https://latex.codecogs.com/png.latex?log%28%5Cfrac%7B%5Ctheta%28x%29%7D%7B1-%5Ctheta%28x%29%7D%29
"log(\\frac{\\theta(x)}{1-\\theta(x)})") is known as odds, which can be
described as “odds in favor of success”

## Likelihood Function

Let ![\\theta(X) =
\\frac{1}{1+e^{-(X\\beta)}}](https://latex.codecogs.com/png.latex?%5Ctheta%28X%29%20%3D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%28X%5Cbeta%29%7D%7D
"\\theta(X) = \\frac{1}{1+e^{-(X\\beta)}}"), then the likelihood
function is the function of the unkown probability of success
![\\theta(X)](https://latex.codecogs.com/png.latex?%5Ctheta%28X%29
"\\theta(X)") given by..

  
![L = \\prod\_{i=1}^{n} P(Y\_i = y\_i|X =
x\_i)](https://latex.codecogs.com/png.latex?L%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7Bn%7D%20P%28Y_i%20%3D%20y_i%7CX%20%3D%20x_i%29
"L = \\prod_{i=1}^{n} P(Y_i = y_i|X = x_i)")  

  
![ = \\prod\_{i=1}^{n} \\binom{m\_i}{y\_i} \\theta(x\_i)^{y\_i}
(1-\\theta(x\_i))^{m\_i - y\_i}
](https://latex.codecogs.com/png.latex?%20%3D%20%5Cprod_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cbinom%7Bm_i%7D%7By_i%7D%20%5Ctheta%28x_i%29%5E%7By_i%7D%20%281-%5Ctheta%28x_i%29%29%5E%7Bm_i%20-%20y_i%7D%20
" = \\prod_{i=1}^{n} \\binom{m_i}{y_i} \\theta(x_i)^{y_i} (1-\\theta(x_i))^{m_i - y_i} ")  

Then, the log likelihood function is given by

  
![log(L) = \\sum\_{i=1}^{n}\[log(\\binom{m\_i}{y\_i}) +
log(\\theta(x\_i)^{y\_i}) + log((1-\\theta(x\_i))^{m\_i - y\_i})\]
\\\\\\ = \\sum\_{i=1}^{n}\[y\_ilog(\\theta(x\_i)) +
(m\_i-y\_i)log(1-\\theta(x\_i)) + log(\\binom{m\_i}{y\_i})\]
](https://latex.codecogs.com/png.latex?log%28L%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Blog%28%5Cbinom%7Bm_i%7D%7By_i%7D%29%20%2B%20log%28%5Ctheta%28x_i%29%5E%7By_i%7D%29%20%2B%20log%28%281-%5Ctheta%28x_i%29%29%5E%7Bm_i%20-%20y_i%7D%29%5D%20%5C%5C%5C%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_ilog%28%5Ctheta%28x_i%29%29%20%2B%20%28m_i-y_i%29log%281-%5Ctheta%28x_i%29%29%20%2B%20log%28%5Cbinom%7Bm_i%7D%7By_i%7D%29%5D%20
"log(L) = \\sum_{i=1}^{n}[log(\\binom{m_i}{y_i}) + log(\\theta(x_i)^{y_i}) + log((1-\\theta(x_i))^{m_i - y_i})] \\\\\\ = \\sum_{i=1}^{n}[y_ilog(\\theta(x_i)) + (m_i-y_i)log(1-\\theta(x_i)) + log(\\binom{m_i}{y_i})] ")  

The all ![m\_i](https://latex.codecogs.com/png.latex?m_i "m_i") equal to
1 in the binary data. Therefore, for binary response variable,

  
![log(L) = \\sum\_{i=1}^{n}\[y\_ilog(\\theta(x\_i)) +
(1-y\_i)log(1-\\theta(x\_i))\]
](https://latex.codecogs.com/png.latex?log%28L%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_ilog%28%5Ctheta%28x_i%29%29%20%2B%20%281-y_i%29log%281-%5Ctheta%28x_i%29%29%5D%20
"log(L) = \\sum_{i=1}^{n}[y_ilog(\\theta(x_i)) + (1-y_i)log(1-\\theta(x_i))] ")  
The constant,
![log(\\binom{m\_i}{y\_i})](https://latex.codecogs.com/png.latex?log%28%5Cbinom%7Bm_i%7D%7By_i%7D%29
"log(\\binom{m_i}{y_i})") is excluded here, since it won’t affect in
optimization.

Then, the cost function is the mean of the Log Likelihood function.

  
![J = \\frac{1}{n}\\sum\_{i=1}^{n}\[y\_ilog(\\theta(x\_i)) +
(1-y\_i)log(1-\\theta(x\_i))\]
](https://latex.codecogs.com/png.latex?J%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5By_ilog%28%5Ctheta%28x_i%29%29%20%2B%20%281-y_i%29log%281-%5Ctheta%28x_i%29%29%5D%20
"J = \\frac{1}{n}\\sum_{i=1}^{n}[y_ilog(\\theta(x_i)) + (1-y_i)log(1-\\theta(x_i))] ")  

## Newton-Raphson Method

It’s often called as Newton’s method, and the form is given by

  
![\\theta\_{k+1} = \\theta\_k -
H\_k^{-1}g\_k](https://latex.codecogs.com/png.latex?%5Ctheta_%7Bk%2B1%7D%20%3D%20%5Ctheta_k%20-%20H_k%5E%7B-1%7Dg_k
"\\theta_{k+1} = \\theta_k - H_k^{-1}g_k")  
where ![H\_k](https://latex.codecogs.com/png.latex?H_k "H_k") is the
Hessian matrix and ![g\_k](https://latex.codecogs.com/png.latex?g_k
"g_k") is gradient matrix.

It comes from the Taylor approximation of
![f(\\theta)](https://latex.codecogs.com/png.latex?f%28%5Ctheta%29
"f(\\theta)") around
![\\theta\_k](https://latex.codecogs.com/png.latex?%5Ctheta_k
"\\theta_k").

  
![f\_{quad}(\\theta) = f(\\theta\_k) + g\_k^{T}(\\theta - \\theta\_k) +
\\frac{1}{2}(\\theta - \\theta\_k)^{T}H\_k(\\theta -
\\theta\_k)](https://latex.codecogs.com/png.latex?f_%7Bquad%7D%28%5Ctheta%29%20%3D%20%20f%28%5Ctheta_k%29%20%2B%20g_k%5E%7BT%7D%28%5Ctheta%20-%20%5Ctheta_k%29%20%2B%20%5Cfrac%7B1%7D%7B2%7D%28%5Ctheta%20-%20%5Ctheta_k%29%5E%7BT%7DH_k%28%5Ctheta%20-%20%5Ctheta_k%29
"f_{quad}(\\theta) =  f(\\theta_k) + g_k^{T}(\\theta - \\theta_k) + \\frac{1}{2}(\\theta - \\theta_k)^{T}H_k(\\theta - \\theta_k)")  

Setting the derivative of the function for
![\\theta](https://latex.codecogs.com/png.latex?%5Ctheta "\\theta") 0.
  
![ \\\\ \\frac{\\partial f\_{quad}(\\theta)}{\\partial \\theta} = 0 +
g\_k + H\_k(\\theta - \\theta\_k) = 0 \\\\ -g\_k =
H\_k(\\theta-\\theta\_k) \\\\ = H\_k\\theta - H\_k\\theta\_k \\\\
H\_k\\theta = H\_k\\theta\_k - g\_k \\\\ \\therefore \\theta\_{k+1} =
\\theta\_k - H\_k^{-1}g\_k
](https://latex.codecogs.com/png.latex?%20%5C%5C%20%5Cfrac%7B%5Cpartial%20f_%7Bquad%7D%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%7D%20%3D%200%20%2B%20g_k%20%2B%20H_k%28%5Ctheta%20-%20%5Ctheta_k%29%20%3D%200%20%5C%5C%20-g_k%20%3D%20H_k%28%5Ctheta-%5Ctheta_k%29%20%5C%5C%20%3D%20H_k%5Ctheta%20-%20H_k%5Ctheta_k%20%5C%5C%20H_k%5Ctheta%20%3D%20H_k%5Ctheta_k%20-%20g_k%20%5C%5C%20%5Ctherefore%20%5Ctheta_%7Bk%2B1%7D%20%3D%20%5Ctheta_k%20-%20H_k%5E%7B-1%7Dg_k%20
" \\\\ \\frac{\\partial f_{quad}(\\theta)}{\\partial \\theta} = 0 + g_k + H_k(\\theta - \\theta_k) = 0 \\\\ -g_k = H_k(\\theta-\\theta_k) \\\\ = H_k\\theta - H_k\\theta_k \\\\ H_k\\theta = H_k\\theta_k - g_k \\\\ \\therefore \\theta_{k+1} = \\theta_k - H_k^{-1}g_k ")  

## Linear Regression with Newton-Raphson method

In linear regression, the ![\\hat
y](https://latex.codecogs.com/png.latex?%5Chat%20y "\\hat y") is given
by ![X\\theta](https://latex.codecogs.com/png.latex?X%5Ctheta
"X\\theta"), and therefore, least squares, which is the cost function,
is given by ..   
![f(\\theta) = f(\\theta; X, y) =
(y-X\\theta)^{T}(y-X\\theta)](https://latex.codecogs.com/png.latex?f%28%5Ctheta%29%20%3D%20f%28%5Ctheta%3B%20X%2C%20y%29%20%3D%20%28y-X%5Ctheta%29%5E%7BT%7D%28y-X%5Ctheta%29
"f(\\theta) = f(\\theta; X, y) = (y-X\\theta)^{T}(y-X\\theta)")  

  
![= y^Ty - y^TX\\theta - \\theta^TX^Ty + \\theta^TX^TX\\theta \\\\ =
y^Ty - 2\\theta^TX^Ty +
\\theta^TX^TX\\theta](https://latex.codecogs.com/png.latex?%3D%20y%5ETy%20-%20y%5ETX%5Ctheta%20-%20%5Ctheta%5ETX%5ETy%20%2B%20%5Ctheta%5ETX%5ETX%5Ctheta%20%5C%5C%20%20%3D%20y%5ETy%20-%202%5Ctheta%5ETX%5ETy%20%2B%20%5Ctheta%5ETX%5ETX%5Ctheta
"= y^Ty - y^TX\\theta - \\theta^TX^Ty + \\theta^TX^TX\\theta \\\\  = y^Ty - 2\\theta^TX^Ty + \\theta^TX^TX\\theta")  
Then, the gradient is..

  
![g = \\partial f(\\theta) = -2X^T y + 2X^T X
\\theta](https://latex.codecogs.com/png.latex?g%20%3D%20%5Cpartial%20f%28%5Ctheta%29%20%3D%20-2X%5ET%20y%20%2B%202X%5ET%20X%20%5Ctheta
"g = \\partial f(\\theta) = -2X^T y + 2X^T X \\theta")  

The Hessian is..   
![H = \\partial^2f(\\theta) = 2X^TX
](https://latex.codecogs.com/png.latex?H%20%3D%20%5Cpartial%5E2f%28%5Ctheta%29%20%3D%202X%5ETX%20
"H = \\partial^2f(\\theta) = 2X^TX ")  

The Newton’s method for Linear Regression is..

  
![\\therefore \\theta\_{k+1} = \\theta\_k - H\_k^{-1}g\_k \\\\ =
\\theta\_k - (2X^TX)^{-1}(-2X^Ty + 2X^TX\\theta\_k) \\\\ = \\theta\_k +
(X^TX)^{-1}(X^Ty) - (X^TX)^{-1}X^TX\\theta\_k \\\\ \\therefore \\hat
\\beta= (X^TX)^{-1}X^Ty
](https://latex.codecogs.com/png.latex?%5Ctherefore%20%5Ctheta_%7Bk%2B1%7D%20%3D%20%5Ctheta_k%20-%20H_k%5E%7B-1%7Dg_k%20%5C%5C%20%3D%20%5Ctheta_k%20-%20%282X%5ETX%29%5E%7B-1%7D%28-2X%5ETy%20%2B%202X%5ETX%5Ctheta_k%29%20%5C%5C%20%3D%20%5Ctheta_k%20%2B%20%28X%5ETX%29%5E%7B-1%7D%28X%5ETy%29%20-%20%28X%5ETX%29%5E%7B-1%7DX%5ETX%5Ctheta_k%20%5C%5C%20%5Ctherefore%20%5Chat%20%5Cbeta%3D%20%28X%5ETX%29%5E%7B-1%7DX%5ETy%20%20
"\\therefore \\theta_{k+1} = \\theta_k - H_k^{-1}g_k \\\\ = \\theta_k - (2X^TX)^{-1}(-2X^Ty + 2X^TX\\theta_k) \\\\ = \\theta_k + (X^TX)^{-1}(X^Ty) - (X^TX)^{-1}X^TX\\theta_k \\\\ \\therefore \\hat \\beta= (X^TX)^{-1}X^Ty  ")  

## Logistic Regression with Newton-Raphson method

the gradient is

  
![X^T(\\theta(X) - y)
](https://latex.codecogs.com/png.latex?X%5ET%28%5Ctheta%28X%29%20-%20y%29%20
"X^T(\\theta(X) - y) ")  
the Hessian is

  
![ X^TWX ](https://latex.codecogs.com/png.latex?%20X%5ETWX%20
" X^TWX ")  
where the ![W](https://latex.codecogs.com/png.latex?W "W") is the
diagonal elements of the matrix,
![\\theta\_k(1-\\theta\_k)](https://latex.codecogs.com/png.latex?%5Ctheta_k%281-%5Ctheta_k%29
"\\theta_k(1-\\theta_k)")

Then,

  
![\\theta\_{k+1} = \\theta\_k - H\_k^{-1}g\_k \\\\ = \\theta\_k -
(X^TWX)^{-1}(X^T(\\theta\_k(X) - y)) \\\\ \\therefore \\hat \\beta =
\\theta\_k - (X^TWX)^{-1}(X^T(\\theta\_k(X) -
y))](https://latex.codecogs.com/png.latex?%5Ctheta_%7Bk%2B1%7D%20%3D%20%5Ctheta_k%20-%20H_k%5E%7B-1%7Dg_k%20%5C%5C%20%3D%20%5Ctheta_k%20-%20%28X%5ETWX%29%5E%7B-1%7D%28X%5ET%28%5Ctheta_k%28X%29%20-%20y%29%29%20%5C%5C%20%5Ctherefore%20%5Chat%20%5Cbeta%20%3D%20%5Ctheta_k%20-%20%28X%5ETWX%29%5E%7B-1%7D%28X%5ET%28%5Ctheta_k%28X%29%20-%20y%29%29
"\\theta_{k+1} = \\theta_k - H_k^{-1}g_k \\\\ = \\theta_k - (X^TWX)^{-1}(X^T(\\theta_k(X) - y)) \\\\ \\therefore \\hat \\beta = \\theta_k - (X^TWX)^{-1}(X^T(\\theta_k(X) - y))")  

``` r
library(numDeriv)
library(MASS)
library(dplyr)
library(caret)
library(titanic)
```

``` r
#Sigmoid function - make cost function derivative
sigmoid <- function(lo){
  sig <- 1/(1+exp(-lo))
  return(sig)
}

#cost function - mean of negative log likelihood function
LL.cost <- function(X,Y,theta){
  
  J <- (-1/n)*sum(Y*log(sigmoid(X%*%theta)) + (1-Y)*log(1-sigmoid(X%*%theta)))
  
  return(J)
}

#first derivatives
gradient.func <- function(X,Y,theta){
  G <- (1/n)*(t(X)%*%(sigmoid(X%*%theta) - Y))
  return(G)
}

#second derivatives
#not use this function
hessian.func <- function(X,Y,theta){
  H <- (1/n)*((t(X)*diag(sigmoid(X%*%theta)*(1-sigmoid(X%*%theta))))%*%X)
  return(H)
}

#Logistic regression with Newton Raphson method
newton.method.logisticReg <- function(X, y, theta, num_iter,tol){
  
  summar <- data.frame(H.diag = rep(0,p+1), theta = rep(0,p+1), se.theta=rep(0,p+1))
  p <- dim(x)[2]
  for(i in 1:num_iter){
    grad <- gradient.func(X,y,theta)
    H <- hessian(LL.cost, X = X, Y = y, theta, method = "complex")
    #"complex" method = calculated as the Jacobian of the gradient
    
    H.diag <- diag(ginv(H))/(n)
    se.theta <- sqrt(abs(H.diag))
    newton.weight <- ginv(H)%*%grad
    theta <- theta - newton.weight
    
    summar$theta <- theta
    
    #new hessian with updated theta
    H.new <- hessian(LL.cost, X = X, Y = y, theta, method = "complex")
    #"complex" method = calculated as the Jacobian of the gradient
    
    H.diag <- diag(ginv(H.new))/(n)
    se.theta <- sqrt(abs(H.diag))
    
    summar$H.diag <- H.diag
    summar$se.theta <- se.theta
    summar$z.value <- summar$theta/summar$se.theta
    summar$p.value <- pnorm(-abs(summar$theta)/summar$se.theta)*2
    summar$lower.conf <- summar$theta - 1.96 * summar$se.theta
    summar$upper.conf <- summar$theta + 1.96 * summar$se.theta
    summar$loglik <- -LL.cost(X,y,summar$theta)*n
    summar$AIC <- -2*summar$loglik + 2*(p+1)
    
    #stop if weights are less than tolerance
    if(any(abs(newton.weight)<tol)){
      summar$iter <- i
      return(summar)
    }
  }
}
```

``` r
set.seed(11)

x <- matrix(rnorm(400), ncol = 4) #random predictors
y <- round(runif(100, 0, 1)) # create a binary outcome
n <- length(x)
X <- cbind(rep(1, 100), x)
p <- dim(x)[2] #intercept term (Beta0) - # of predictors
theta<-rep(0, 5)

data1 <- data.frame(cbind(y,x))

result1 <- newton.method.logisticReg(X,y,theta, 20,1e-5)
result1
```

    ##       H.diag       theta  se.theta    z.value    p.value lower.conf
    ## 1 0.04334712  0.16432469 0.2081997  0.7892647 0.42995730 -0.2437467
    ## 2 0.05126367 -0.08414919 0.2264148 -0.3716594 0.71014649 -0.5279222
    ## 3 0.04397323  0.08311151 0.2096979  0.3963392 0.69185485 -0.3278965
    ## 4 0.04780654 -0.37630974 0.2186471 -1.7210829 0.08523578 -0.8048580
    ## 5 0.04223175  0.03286426 0.2055037  0.1599206 0.87294364 -0.3699229
    ##   upper.conf    loglik      AIC iter
    ## 1 0.57239611 -67.14336 144.2867    3
    ## 2 0.35962386 -67.14336 144.2867    3
    ## 3 0.49411947 -67.14336 144.2867    3
    ## 4 0.05223852 -67.14336 144.2867    3
    ## 5 0.43565144 -67.14336 144.2867    3

``` r
#fitting by glm syntax built in R
glm1 <- glm(y~., data= data1, family = "binomial")

summary(glm1)
```

    ## 
    ## Call:
    ## glm(formula = y ~ ., family = "binomial", data = data1)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -1.578  -1.209   0.873   1.076   1.485  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)  
    ## (Intercept)  0.16432    0.20820   0.789   0.4300  
    ## V2          -0.08415    0.22641  -0.372   0.7101  
    ## V3           0.08311    0.20970   0.396   0.6919  
    ## V4          -0.37631    0.21865  -1.721   0.0852 .
    ## V5           0.03286    0.20550   0.160   0.8729  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 137.63  on 99  degrees of freedom
    ## Residual deviance: 134.29  on 95  degrees of freedom
    ## AIC: 144.29
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
summary(glm1)$coefficients
```

    ##                Estimate Std. Error    z value   Pr(>|z|)
    ## (Intercept)  0.16432469  0.2081997  0.7892648 0.42995727
    ## V2          -0.08414919  0.2264148 -0.3716594 0.71014646
    ## V3           0.08311151  0.2096979  0.3963392 0.69185483
    ## V4          -0.37630974  0.2186470 -1.7210831 0.08523574
    ## V5           0.03286426  0.2055036  0.1599206 0.87294363

``` r
#test 
summary(glm1)$coefficients[,1] == result1$theta
```

    ##       [,1]
    ## [1,] FALSE
    ## [2,] FALSE
    ## [3,] FALSE
    ## [4,] FALSE
    ## [5,] FALSE

``` r
abs(summary(glm1)$coefficients[,1] - result1$theta) < 1e-5
```

    ##      [,1]
    ## [1,] TRUE
    ## [2,] TRUE
    ## [3,] TRUE
    ## [4,] TRUE
    ## [5,] TRUE

``` r
abs(summary(glm1)$coefficients[,2] - result1$se.theta) < 1e-5
```

    ## (Intercept)          V2          V3          V4          V5 
    ##        TRUE        TRUE        TRUE        TRUE        TRUE

``` r
abs(summary(glm1)$coefficients[,3] - result1$z.value) < 1e-5
```

    ##      [,1]
    ## [1,] TRUE
    ## [2,] TRUE
    ## [3,] TRUE
    ## [4,] TRUE
    ## [5,] TRUE

``` r
abs(summary(glm1)$coefficients[,4] - result1$p.value) < 1e-5
```

    ##      [,1]
    ## [1,] TRUE
    ## [2,] TRUE
    ## [3,] TRUE
    ## [4,] TRUE
    ## [5,] TRUE

``` r
abs(confint.default(glm1) - cbind(result1$lower.conf, result1$upper.conf)) < 1e-5
```

    ##             2.5 % 97.5 %
    ## (Intercept)  TRUE   TRUE
    ## V2           TRUE   TRUE
    ## V3           TRUE   TRUE
    ## V4           TRUE   TRUE
    ## V5           TRUE   TRUE

``` r
abs(logLik(glm1) - result1$loglik) < 1e-5
```

    ## [1] TRUE TRUE TRUE TRUE TRUE

``` r
abs(AIC(glm1) - result1$AIC) < 1e-5
```

    ## [1] TRUE TRUE TRUE TRUE TRUE

``` r
#fitting - prediction
pred <- as.vector(sigmoid(X%*%result1$theta))
pred.hands <- ifelse(pred > .5, 1, 0)

pred1 <- as.vector(predict(glm1, data1, type="response"))
pred1.hands <- ifelse(pred1 > .5, 1, 0)

#same results
identical(pred.hands, pred1.hands)
```

    ## [1] TRUE

``` r
#Accuracy
mean(pred.hands == y)
```

    ## [1] 0.58

``` r
mean(pred1.hands == y)
```

    ## [1] 0.58

``` r
summar <- summary(glm1)
#p-value for deviance to test the hypothesis if the model is appropriate
1-pchisq(summar$deviance, df = summar$df.residual)
```

    ## [1] 0.00496655

``` r
1-pchisq(summar$null.deviance - summar$deviance, df = summar$df.null - summar$df.residual)
```

    ## [1] 0.5024556

``` r
#pseudo R^2 for logistic regression
1-summar$deviance/summar$null.deviance
```

    ## [1] 0.02427594

``` r
train<-titanic_train

train <- train %>%
  select(subset= -c(PassengerId, Pclass, Name, Sex, Ticket, Cabin, Embarked)) %>%
  mutate_all(funs(ifelse(is.na(.), mean(., na.rm=TRUE), .))) %>%
  mutate_if(is.integer, as.numeric)
```

    ## Warning: funs() is soft deprecated as of dplyr 0.8.0
    ## please use list() instead
    ## 
    ##   # Before:
    ##   funs(name = f(.))
    ## 
    ##   # After: 
    ##   list(name = ~ f(.))
    ## This warning is displayed once per session.

``` r
train.indx <- createDataPartition(train$Survived, p=0.7, list=FALSE)

training <- train[train.indx,]
valid <- train[-train.indx,]

x <- as.matrix(training %>% select(-Survived))
y <- training$Survived %>% as.numeric
n <- nrow(x)
X <- cbind(rep(1, nrow(x)), x)
p <- dim(x)[2]
theta<-rep(0, ncol(X))
data1 <- data.frame(cbind(y,x))

result1 <- newton.method.logisticReg(X, y, theta,1000, 1e-15)
result1
```

    ##         H.diag       theta    se.theta    z.value      p.value  lower.conf
    ## 1 6.349632e-02  0.03805791 0.251984762  0.1510326 8.799500e-01 -0.45583222
    ## 2 6.324856e-05 -0.03611018 0.007952896 -4.5405074 5.611900e-06 -0.05169786
    ## 3 1.153837e-02 -0.28748460 0.107416812 -2.6763465 7.442966e-03 -0.49802156
    ## 4 1.389262e-02  0.09658128 0.117866949  0.8194094 4.125529e-01 -0.13443794
    ## 5 1.204606e-05  0.02127766 0.003470744  6.1305773 8.756076e-10  0.01447500
    ##    upper.conf    loglik      AIC iter
    ## 1  0.53194804 -374.4383 758.8767    7
    ## 2 -0.02052251 -374.4383 758.8767    7
    ## 3 -0.07694765 -374.4383 758.8767    7
    ## 4  0.32760050 -374.4383 758.8767    7
    ## 5  0.02808032 -374.4383 758.8767    7

``` r
glm1 <- glm(y~x, family = binomial)
summary(glm1)
```

    ## 
    ## Call:
    ## glm(formula = y ~ x, family = binomial)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -3.0151  -0.8933  -0.7737   1.1676   1.9235  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  0.038058   0.251985   0.151  0.87995    
    ## xAge        -0.036110   0.007953  -4.541 5.61e-06 ***
    ## xSibSp      -0.287485   0.107417  -2.676  0.00744 ** 
    ## xParch       0.096581   0.117867   0.819  0.41255    
    ## xFare        0.021278   0.003471   6.131 8.76e-10 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 826.65  on 623  degrees of freedom
    ## Residual deviance: 748.88  on 619  degrees of freedom
    ## AIC: 758.88
    ## 
    ## Number of Fisher Scoring iterations: 5

``` r
#test
summary(glm1)$coefficients[,1] == result1$theta
```

    ##       [,1]
    ## [1,] FALSE
    ## [2,] FALSE
    ## [3,] FALSE
    ## [4,] FALSE
    ## [5,] FALSE

``` r
abs(summary(glm1)$coefficients[,1] - result1$theta) < 1e-5
```

    ##      [,1]
    ## [1,] TRUE
    ## [2,] TRUE
    ## [3,] TRUE
    ## [4,] TRUE
    ## [5,] TRUE

``` r
abs(summary(glm1)$coefficients[,2] - result1$se.theta) < 1e-5
```

    ## (Intercept)        xAge      xSibSp      xParch       xFare 
    ##        TRUE        TRUE        TRUE        TRUE        TRUE

``` r
abs(summary(glm1)$coefficients[,3] - result1$z.value) < 1e-5
```

    ##      [,1]
    ## [1,] TRUE
    ## [2,] TRUE
    ## [3,] TRUE
    ## [4,] TRUE
    ## [5,] TRUE

``` r
abs(summary(glm1)$coefficients[,4] - result1$p.value) < 1e-5
```

    ##      [,1]
    ## [1,] TRUE
    ## [2,] TRUE
    ## [3,] TRUE
    ## [4,] TRUE
    ## [5,] TRUE

``` r
abs(confint.default(glm1) - cbind(result1$lower.conf, result1$upper.conf)) < 1e-5
```

    ##             2.5 % 97.5 %
    ## (Intercept)  TRUE   TRUE
    ## xAge         TRUE   TRUE
    ## xSibSp       TRUE   TRUE
    ## xParch       TRUE   TRUE
    ## xFare        TRUE   TRUE

``` r
abs(logLik(glm1) - result1$loglik) < 1e-5
```

    ## [1] TRUE TRUE TRUE TRUE TRUE

``` r
abs(AIC(glm1) - result1$AIC) < 1e-5
```

    ## [1] TRUE TRUE TRUE TRUE TRUE

``` r
#fitting
pred <- as.vector(sigmoid(X%*%result1$theta))
pred.hands <- ifelse(pred > .5, 1, 0)

pred1 <- as.vector(predict(glm1, data1, type="response"))
pred1.hands <- ifelse(pred1 > .5, 1, 0)

identical(pred.hands, pred1.hands)
```

    ## [1] TRUE

``` r
#Training Accuracy
mean(pred.hands == y)
```

    ## [1] 0.713141

``` r
mean(pred1.hands == y)
```

    ## [1] 0.713141

``` r
#Validation Accuracy
x <- as.matrix(valid %>% select(-Survived))
y <- valid$Survived %>% as.numeric
n <- nrow(x)
X <- cbind(rep(1, nrow(x)), x)
p <- dim(x)[2]

data1 <- data.frame(cbind(y,x))

pred <- as.vector(sigmoid(X%*%result1$theta))
pred.hands <- ifelse(pred > .5, 1, 0)

pred1 <- as.vector(predict(glm1, data1, type="response"))
pred1.hands <- ifelse(pred1 > .5, 1, 0)

#Validation Accuracy
mean(pred.hands == y)
```

    ## [1] 0.6816479

``` r
mean(pred1.hands == y)
```

    ## [1] 0.6816479

``` r
summar <- summary(glm1)
#p-value for deviance to test the hypothesis if the model is appropriate
1-pchisq(summar$deviance, df = summar$df.residual)
```

    ## [1] 0.0002513425

``` r
1-pchisq(summar$null.deviance - summar$deviance, df = summar$df.null - summar$df.residual)
```

    ## [1] 5.551115e-16

``` r
#R^2
1-summar$deviance/summar$null.deviance
```

    ## [1] 0.09407787

``` r
#removing one of the classes of the target variable, virginica, 
#to make target as a binary variable
iris1 <- iris %>% 
  filter(!Species %in% c("setosa")) %>%
  mutate(Species = as.numeric(Species)-2)

iris1 %>%  cor
```

    ##              Sepal.Length Sepal.Width Petal.Length Petal.Width   Species
    ## Sepal.Length    1.0000000   0.5538548    0.8284787   0.5937094 0.4943049
    ## Sepal.Width     0.5538548   1.0000000    0.5198023   0.5662025 0.3080798
    ## Petal.Length    0.8284787   0.5198023    1.0000000   0.8233476 0.7864237
    ## Petal.Width     0.5937094   0.5662025    0.8233476   1.0000000 0.8281293
    ## Species         0.4943049   0.3080798    0.7864237   0.8281293 1.0000000

``` r
iris1 <- iris1 %>%
  select(-Petal.Length)

train.indx <- createDataPartition(iris1$Species, p=0.8, list=FALSE)

training <- iris1[train.indx,]
valid <- iris1[-train.indx,]


x <- as.matrix(training %>% select(-Species))
y <- training$Species
n <- nrow(x)
X <- cbind(rep(1, nrow(x)), x)
p <- dim(x)[2]
theta<-rep(0, ncol(X))
data1 <- data.frame(cbind(y,x))

result1 <- newton.method.logisticReg(X, y, theta,100, 1e-5)
result1
```

    ##       H.diag      theta  se.theta   z.value     p.value  lower.conf
    ## 1 125.206057 -26.977004 11.189551 -2.410910 0.015912764 -48.9085240
    ## 2   1.640926   1.647647  1.280986  1.286233 0.198361727  -0.8630863
    ## 3   7.123311  -5.475747  2.668953 -2.051646 0.040204093 -10.7068949
    ## 4  39.525382  19.628029  6.286922  3.122041 0.001796017   7.3056624
    ##   upper.conf    loglik     AIC iter
    ## 1 -5.0454831 -9.709402 27.4188    9
    ## 2  4.1583805 -9.709402 27.4188    9
    ## 3 -0.2445985 -9.709402 27.4188    9
    ## 4 31.9503947 -9.709402 27.4188    9

``` r
glm1 <- glm(y~x, family=binomial)
summary(glm1)$coefficient
```

    ##                 Estimate Std. Error   z value    Pr(>|z|)
    ## (Intercept)   -26.977003  11.188397 -2.411159 0.015901920
    ## xSepal.Length   1.647647   1.280896  1.286324 0.198329906
    ## xSepal.Width   -5.475747   2.668743 -2.051807 0.040188405
    ## xPetal.Width   19.628028   6.286361  3.122320 0.001794321

``` r
summary(glm1)
```

    ## 
    ## Call:
    ## glm(formula = y ~ x, family = binomial)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -1.62185  -0.07892  -0.00125   0.04544   2.26573  
    ## 
    ## Coefficients:
    ##               Estimate Std. Error z value Pr(>|z|)   
    ## (Intercept)    -26.977     11.188  -2.411  0.01590 * 
    ## xSepal.Length    1.648      1.281   1.286  0.19833   
    ## xSepal.Width    -5.476      2.669  -2.052  0.04019 * 
    ## xPetal.Width    19.628      6.286   3.122  0.00179 **
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 110.904  on 79  degrees of freedom
    ## Residual deviance:  19.419  on 76  degrees of freedom
    ## AIC: 27.419
    ## 
    ## Number of Fisher Scoring iterations: 8

``` r
confint.default(glm1)
```

    ##                     2.5 %     97.5 %
    ## (Intercept)   -48.9058591 -5.0481477
    ## xSepal.Length  -0.8628621  4.1581563
    ## xSepal.Width  -10.7063874 -0.2451059
    ## xPetal.Width    7.3069868 31.9490701

``` r
logLik(glm1)
```

    ## 'log Lik.' -9.709402 (df=4)

``` r
pred <- as.vector(sigmoid(X%*%result1$theta))
pred.hands <- ifelse(pred > .5, 1, 0)

pred1 <- as.vector(predict(glm1, data1, type="response"))
pred1.hands <- ifelse(pred1 > .5, 1, 0)

identical(pred.hands, pred1.hands)
```

    ## [1] TRUE

``` r
#Training Accuracy
mean(pred.hands == y)
```

    ## [1] 0.9375

``` r
mean(pred1.hands == y)
```

    ## [1] 0.9375

``` r
x <- as.matrix(valid %>% select(-Species))
y <- valid$Species
n <- nrow(x)
X <- cbind(rep(1, nrow(x)), x)
p <- dim(x)[2]

data1 <- data.frame(cbind(y,x))


pred <- as.vector(sigmoid(X%*%result1$theta))
pred.hands <- ifelse(pred > .5, 1, 0)

pred1 <- as.vector(predict(glm1, data1, type="response"))
pred1.hands <- ifelse(pred1 > .5, 1, 0)

identical(pred.hands, pred1.hands)
```

    ## [1] TRUE

``` r
#Training Accuracy
mean(pred.hands == y)
```

    ## [1] 0.95

``` r
mean(pred1.hands == y)
```

    ## [1] 0.95

``` r
summar <- summary(glm1)
#p-value for deviance to test the hypothesis if the model is appropriate
1-pchisq(summar$deviance, df = summar$df.residual)
```

    ## [1] 1

``` r
1-pchisq(summar$null.deviance - summar$deviance, df = summar$df.null - summar$df.residual)
```

    ## [1] 0

``` r
#R^2
1-summar$deviance/summar$null.deviance
```

    ## [1] 0.8249037
