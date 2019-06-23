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


E1 = zeros(1000, 1);
E2 = zeros(1000, 1);

W11 = 2*rand(4, 3) - 1;      % Cross entropy （交叉熵）   初始化权重（第一层）   
W12 = 2*rand(1, 4) - 1;      %                             第二层 
W21 = W11;                   % Sum of squared error（误差平方函数）
W22 = W12;                   %

for epoch = 1:1000               %训练轮数
  [W11 W12] = BackpropCE(W11, W12, X, D);          
  [W21 W22] = BackpropXOR(W21, W22, X, D);

  es1 = 0;
  es2 = 0;
  N   = 4;
  for k = 1:N
    x = X(k, :)';
    d = D(k);

    v1  = W11*x;
    y1  = Sigmoid(v1);
    v   = W12*y1;
    y   = Sigmoid(v);
    es1 = es1 + (d - y)^2;        %求每组数误差之和
    
    v1  = W21*x;
    y1  = Sigmoid(v1);
    v   = W22*y1;
    y   = Sigmoid(v);
    es2 = es2 + (d - y)^2;
  end
  E1(epoch) = es1 / N;      %第epoch轮的误差，输入到E1中的第epoch个元素
  E2(epoch) = es2 / N;
end

plot(E1, 'r')
hold on
plot(E2, 'b:')
xlabel('Epoch')
ylabel('Average of Training error')
legend('Cross Entropy', 'Sum of Squared Error')

