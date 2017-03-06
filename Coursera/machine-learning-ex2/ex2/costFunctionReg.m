function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
gOfz = sigmoid(z);
temp = theta;
temp(1) = 0;
J = (1/m) * sum(-y .* log(gOfz) .- (1 .- y) .* log(1 .- gOfz)) + (lambda/(2*m)) * sum(temp.^2);
%temp = theta;
%temp(1) = 0;
%grad([;2:end]) = (1/m) .* X(:,2:end)' * (gOfz .- y) .+ (lambda/m) * theta([;2:end]);
grad = 1/m * X' * (gOfz - y) + lambda/m * temp;
%grad(1) = (1/m)* X(:,1)' * (gOfz - y);
%grad = (1/m) * X' * (gOfz - y) + (lambda/m) * temp;

%one = ones(m,1);
%n = size(X,2); % number of features
%g = sigmoid(X * theta);
%grad = 1/m * X' * (sigmoid(X * theta) - y) + lambda/m * theta;
%grad(1) = 1/m * X(:,1)' * (g - y); %dont add lamda/m *theta to grad0

%J = 1/m * ( ((log(g))' * -y) - ((one - y)'  * (log((one - g))))) + lambda/(2*m) * sum(theta(2:n,1).^2);





% =============================================================

end
