function [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D)
%
%���򴫲��޸�Ȩ�أ���������˲�����

alpha = 0.01;
beta  = 0.95;    %������������

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D);             %������֮�����ֵ���˴�ѵ��8000�����ݣ�8000��������

bsize = 100;                    %��С�����ݶ��½��㷨  ÿ��ѵ��һ�������ݣ��޸�Ȩ��
blist = 1:bsize:(N-bsize+1);    %��һ��80���޸�Ȩ�أ�blist����ÿһС�����ݵ��һ���������

% One epoch loop ��һ��ѭ��
%
for batch = 1:length(blist)     % batch=1:80 ���е���80�Σ�ÿһ���У�
  dW1 = zeros(size(W1));
  dW5 = zeros(size(W5));
  dWo = zeros(size(Wo));
  
  % Mini-batch loop      һ��ѵ���е�С�����㷨
  %
  begin = blist(batch);
  for k = begin:begin+bsize-1      %begin:begin+99
    % Forward pass = inference ����������⣩
    %
    x  = X(:, :, k);               % Input,           28x28  ����ͼ��
    y1 = Conv(x, W1);              % Convolution,  20x20x20  ���������
                                   %��9*9����˲�������ͼ������20*20������ͼ������ӳ�䣩����20������˲���������20������
    y2 = ReLU(y1);                 %
    y3 = Pool(y2);                 % Pooling,      10x10x20
                                   %��2*2��ƽ���ػ�����20������ӳ������Ϊ20��10*10��ӳ��
    y4 = reshape(y3, [], 1);       %��y3����Ϊ������ά��Ϊ20*10*10=2000
    v5 = W5*y4;                    % ReLU,             2000
    y5 = ReLU(v5);                 %
    v  = Wo*y5;                    % Softmax,          10x1
    y  = Softmax(v);               %                       ���y��ֵ

    % One-hot encoding
    %
    d = zeros(10, 1);
   
    %d(D(k))=1;        %xi �����Դ���d(sub2ind(size(d), D(k),1))=1;�˴�D(k)Ϊ�������±�������������һ��
    d(sub2ind(size(d), D(k), 1)) = 1;        %sub2ind ��Ԫ���±�ת��Ϊ����������

    % Backpropagation   ���򴫲������
    %
    e      = d - y;                   % Output layer  
    delta  = e;

    e5     = Wo' * delta;             % Hidden(ReLU) layer
    delta5 = (y5 > 0) .* e5;

    e4     = W5' * delta5;           
    
    e3     = reshape(e4, size(y3));    % Pooling layer  �ػ��㣨���С��㼯�ĺ��壩

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
  
  % Update weights  ����Ȩ��
  %
  dW1 = dW1 / bsize;
  dW5 = dW5 / bsize;
  dWo = dWo / bsize;
  
  momentum1 = alpha*dW1 + beta*momentum1;    %���������㷨
  W1        = W1 + momentum1;
  
  momentum5 = alpha*dW5 + beta*momentum5;
  W5        = W5 + momentum5;
   
  momentumo = alpha*dWo + beta*momentumo;
  Wo        = Wo + momentumo;  
end

end

