clear all
           
X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0
      1
      1
      0
    ];
n=4   %n为隐藏层节点数

W1 = 2*rand(n, 3) - 1;   %初始化权值  4为隐藏层节点数
W2 = 2*rand(1, n) - 1;

for epoch = 1:10000          % train  训练次数
  [W1 W2] = BackpropXOR(W1, W2, X, D);    %训练函数（反向传播）
end

N = 4;                        % inference  N为输入的数据维度为4，需要训练4组数据
for k = 1:N
  x  = X(k, :)';
  v1 = W1*x;
  y1 = Sigmoid(v1);
  v  = W2*y1;
  y  = Sigmoid(v)
end