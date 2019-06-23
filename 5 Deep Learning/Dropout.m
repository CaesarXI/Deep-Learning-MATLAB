function ym = Dropout(y, ratio)
  [m, n] = size(y);  
  ym     = zeros(m, n);
  
  num     = round(m*n*(1-ratio));      %round��������������   numΪ����Ԫ�ظ���
  idx     = randperm(m*n, num);        %ȡ������Ԫ�ص�����
  ym(idx) = 1 / (1-ratio);             %��ֵ������Ԫ��
end