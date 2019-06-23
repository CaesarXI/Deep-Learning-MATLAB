function [W1, W2] = BackpropCE(W1, W2, X, D)
  alpha = 0.9;
  
  N = 4;      % N为数据组的个数（一组包括输入输出）
  for k = 1:N
    x = X(k, :)';        % x = a column vector
    d = D(k);
    
    v1 = W1*x;
    y1 = Sigmoid(v1);    
    v  = W2*y1;
    y  = Sigmoid(v);
    
    e     = d - y;            %误差
    delta = e;               %输出层偏差

    e1     = W2'*delta;
    delta1 = y1.*(1-y1).*e1;    %隐藏层偏差
    
    dW1 = alpha*delta1*x';
    W1 = W1 + dW1;
    
    dW2 = alpha*delta*y1';    
    W2 = W2 + dW2;
  end
end