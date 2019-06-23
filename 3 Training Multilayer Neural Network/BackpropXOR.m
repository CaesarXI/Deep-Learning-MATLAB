function [W1, W2] = BackpropXOR(W1, W2, X, D)
  alpha = 0.9;  %学习率  W1 W2为对应层的权重矩阵
  
  N = 4;    
  for k = 1:N
    x = X(k, :)';   %输入
    d = D(k);   %正确输出
    
    %前向训练，求解节点输出
    v1 = W1*x;
    y1 = Sigmoid(v1);    
    v  = W2*y1;
    y  = Sigmoid(v);
    
    e     = d - y;     %误差
    delta = y.*(1-y).*e;     %反向求增量 delta           %向量元素乘积运算 .*（加点）所有元素进行乘积计算

    e1     = W2'*delta;
    delta1 = y1.*(1-y1).*e1;   %隐藏层增量 delta1
    
    dW1 = alpha*delta1*x';    %权重更新
    W1  = W1 + dW1;
    
    dW2 = alpha*delta*y1';    
    W2  = W2 + dW2;
  end
end