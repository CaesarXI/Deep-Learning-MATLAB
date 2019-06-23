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
n=4   %nΪ���ز�ڵ���

W1 = 2*rand(n, 3) - 1;   %��ʼ��Ȩֵ  4Ϊ���ز�ڵ���
W2 = 2*rand(1, n) - 1;

for epoch = 1:10000          % train  ѵ������
  [W1 W2] = BackpropXOR(W1, W2, X, D);    %ѵ�����������򴫲���
end

N = 4;                        % inference  NΪ���������ά��Ϊ4����Ҫѵ��4������
for k = 1:N
  x  = X(k, :)';
  v1 = W1*x;
  y1 = Sigmoid(v1);
  v  = W2*y1;
  y  = Sigmoid(v)
end