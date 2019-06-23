function y = Pool(x)
% ƽ���ػ�������    
% 2x2 mean pooling
%
%
[xrow, xcol, numFilters] = size(x);

y = zeros(xrow/2, xcol/2, numFilters);
for k = 1:numFilters
  filter = ones(2) / (2*2);    % for mean    ���������˲�����ʵ��ƽ���ػ���
  image  = conv2(x(:, :, k), filter, 'valid');          %��ά�������conv2
                                                      %�ػ�����Ҳ��һ�־������
  y(:, :, k) = image(1:2:end, 1:2:end);
end

end
 