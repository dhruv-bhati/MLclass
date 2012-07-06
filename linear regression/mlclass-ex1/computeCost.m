function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


temp = X * theta;
temp1 = temp - y;
temp2 = temp1 .^ 2;
temp3 = sum(temp2);
temp4 = temp3/(2 * m);
J = temp4;


% =========================================================================

end