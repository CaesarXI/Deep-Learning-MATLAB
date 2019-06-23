function [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D)
%
%反向传播修改权重（包括卷积核参数）

alpha = 0.01;
beta  = 0.95;    %常量：动量法

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D);             %求行列之中最大值，此处训练8000组数据，8000个输出标记

bsize = 100;                    %做小批量梯度下降算法  每次训练一百组数据，修改权重
blist = 1:bsize:(N-bsize+1);    %第一轮80次修改权重，blist包含每一小批数据点第一个点的索引

% One epoch loop 第一轮循环
%
for batch = 1:length(blist)     % batch=1:80 进行调整80次（每一轮中）
  dW1 = zeros(size(W1));
  dW5 = zeros(size(W5));
  dWo = zeros(size(Wo));
  
  % Mini-batch loop      一轮训练中的小批量算法
  %
  begin = blist(batch);
  for k = begin:begin+bsize-1      %begin:begin+99
    % Forward pass = inference 推理（正向求解）
    %
    x  = X(:, :, k);               % Input,           28x28  输入图像
    y1 = Conv(x, W1);              % Convolution,  20x20x20  卷积操作，
                                   %用9*9卷积滤波器处理图像，生产20*20的特征图像（特征映射），有20个卷积滤波器，生成20个特征
    y2 = ReLU(y1);                 %
    y3 = Pool(y2);                 % Pooling,      10x10x20
                                   %用2*2的平均池化处理，20个特征映射缩减为20个10*10的映射
    y4 = reshape(y3, [], 1);       %将y3整理为向量，维度为20*10*10=2000
    v5 = W5*y4;                    % ReLU,             2000
    y5 = ReLU(v5);                 %
    v  = Wo*y5;                    % Softmax,          10x1
    y  = Softmax(v);               %                       解出y的值

    % One-hot encoding
    %
    d = zeros(10, 1);
   
    %d(D(k))=1;        %xi ：可以代替d(sub2ind(size(d), D(k),1))=1;此处D(k)为向量，下标与线性索引号一致
    d(sub2ind(size(d), D(k), 1)) = 1;        %sub2ind 将元素下标转换为线性索引号

    % Backpropagation   反向传播求误差
    %
    e      = d - y;                   % Output layer  
    delta  = e;

    e5     = Wo' * delta;             % Hidden(ReLU) layer
    delta5 = (y5 > 0) .* e5;

    e4     = W5' * delta5;           
    
    e3     = reshape(e4, size(y3));    % Pooling layer  池化层（集中、汇集的含义）

    e2 = zeros(size(y2));           
    W3 = ones(size(y2)) / (2*2);
    for c = 1:20
      e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
    end
    
    delta2 = (y2 > 0) .* e2;          % ReLU layer
  
    delta1_x = zeros(size(W1));       % Convolutional layer
    for c = 1:20
      delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
    end
    
    dW1 = dW1 + delta1_x; 
    dW5 = dW5 + delta5*y4';    
    dWo = dWo + delta *y5';
  end 
  
  % Update weights  更新权重
  %
  dW1 = dW1 / bsize;
  dW5 = dW5 / bsize;
  dWo = dWo / bsize;
  
  momentum1 = alpha*dW1 + beta*momentum1;    %动量更新算法
  W1        = W1 + momentum1;
  
  momentum5 = alpha*dW5 + beta*momentum5;
  W5        = W5 + momentum5;
   
  momentumo = alpha*dWo + beta*momentumo;
  Wo        = Wo + momentumo;  
end

end

