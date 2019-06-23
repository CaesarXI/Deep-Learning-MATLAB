function y = Conv(x, W)
%Conv������������ͼ��;�����������󣬷�������ӳ��
%

[wrow, wcol, numFilters] = size(W);
[xrow, xcol, ~         ] = size(x);

yrow = xrow - wrow + 1;              %�����Ĵ�С
ycol = xcol - wcol + 1;

y = zeros(yrow, ycol, numFilters);

for k = 1:numFilters
  filter = W(:, :, k); 
  filter = rot90(squeeze(filter), 2);         %�淽����ת����ͼƬ��2*90��
  y(:, :, k) = conv2(x, filter, 'valid');     %conv2ΪMATLAB���õľ������
end

end

