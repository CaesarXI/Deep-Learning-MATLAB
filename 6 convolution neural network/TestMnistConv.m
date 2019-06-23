clear all

%读取数据集
Images = loadMNISTImages('./MNIST/t10k-images.idx3-ubyte');      
Images = reshape(Images, 28, 28, []);
Labels = loadMNISTLabels('./MNIST/t10k-labels.idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10   若标签值为0则修改为10


rng(1);

% Learning             训练网络
%
W1 = 1e-2*randn([9 9 20]);         %9*9的卷积滤波器，共有20个卷积滤波器，生产20个特征
W5 = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
Wo = (2*rand( 10,  100) - 1) * sqrt(6) / sqrt( 10 +  100);

X = Images(:, :, 1:8000);
D = Labels(1:8000);

for epoch = 1:3
  epoch            %输出此时进行的轮数
  [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D);
end

save('MnistConv.mat');     %将网络参数保存


% Test       测试网络
%
X = Images(:, :, 8001:10000);
D = Labels(8001:10000);

acc = 0;
N   = length(D);
for k = 1:N
  x = X(:, :, k);                   % Input,           28x28

  y1 = Conv(x, W1);                 % Convolution,  20x20x20
  y2 = ReLU(y1);                    %
  y3 = Pool(y2);                    % Pool,         10x10x20
  y4 = reshape(y3, [], 1);          %                   2000  
  v5 = W5*y4;                       % ReLU,              360
  y5 = ReLU(v5);                    %
  v  = Wo*y5;                       % Softmax,            10
  y  = Softmax(v);                  %

  [~, i] = max(y);                  % 最大值索引  [Y,U]=max(A)：返回行向量Y和U，Y向量记录A的每列的最大值，U向量记录每列最大值的行号。
                                    % ~ 符号表示占位符，表示忽略该位置元素
  if i == D(k)                      % 将神经网络的模型输出与正确输出比较，计算其匹配个数，得到准确率
    acc = acc + 1;
  end
end

acc = acc / N;                              
fprintf('Accuracy is %f\n', acc);


