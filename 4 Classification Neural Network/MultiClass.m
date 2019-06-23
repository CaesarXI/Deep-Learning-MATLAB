function [W1, W2] = MultiClass(W1, W2, X, D)
  alpha = 0.9;
  
  N = 5;  
  for k = 1:N
    x = reshape(X(:, :, k), 25, 1);   %产生25维列向量，取第k层    reshape函数对5*5矩阵形式整理为25*1的列向量
    d = D(k, :)';
    
    v1 = W1*x;
    y1 = Sigmoid(v1);
    v  = W2*y1;
    y  = Softmax(v);
    
    e     = d - y;
    delta = e;         %输出层，增量delta

    e1     = W2'*delta;
    delta1 = y1.*(1-y1).*e1;        %隐藏层增量
    
    dW1 = alpha*delta1*x';
    W1 = W1 + dW1;
    
    dW2 = alpha*delta*y1';   
    W2 = W2 + dW2;
  end
end