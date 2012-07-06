function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


temp = X * theta;
temp1 = temp - y;
temp2 = temp1 .^ 2;
temp3 = sum(temp2);
temp4 = temp3/(2 * m);
J = temp4;


thet = (theta .^ 2)';
s=sum(thet)-theta(1,1)*theta(1,1);
p=(lambda/(2*m))*s;
J=J+p;


a=(temp1.*X(:,1))';
n=sum(a);
n=(1/m)*n;
grad(1,1)=n;
for j=2:size(theta)
a=(temp1.*X(:,j))';
n=sum(a);
n=(1/m)*n;
n=n+(lambda/m)*theta(j,1);
grad(j,1)=n;
endfor








% =========================================================================

grad = grad(:);

end
