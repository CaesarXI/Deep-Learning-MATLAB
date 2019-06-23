clear all
           
X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0
      0
      1
      1
    ];
      
W = 2*rand(1, 3) - 1;   %初始化权重 rand为1*3的矩阵，每个元素在0到1之间随机取值

for epoch = 1:10000           % train
  W = DeltaSGD(W, X, D);
end

N = 4;                        % inference
for k = 1:N
  x = X(k, :)';
  v = W*x;
  y = Sigmoid(v)
end