function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    tempTheta1 = theta(1);
    tempTheta2 = theta(2);

    sumTheta1 = 0;
    for i = 1:m
        sumTheta1 = sumTheta1 + ((tempTheta1 + tempTheta2 * X(i, 2)) - y(i)) * X(i, 1);
    end
    theta(1) = tempTheta1 - alpha * (1/m) * sumTheta1;

    sumTheta2 = 0;
    for i = 1:m
        sumTheta2 = sumTheta2 + ((tempTheta1 + tempTheta2 * X(i, 2)) - y(i)) * X(i, 2);
    end
    theta(2) = tempTheta2 - alpha * (1/m) * sumTheta2;






    computeCost(X, y, theta);
    % dbstop(40);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
