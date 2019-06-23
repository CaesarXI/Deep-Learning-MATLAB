function y = Pool(x)
% 平均池化处理函数    
% 2x2 mean pooling
%
%
[xrow, xcol, numFilters] = size(x);

y = zeros(xrow/2, xcol/2, numFilters);
for k = 1:numFilters
  filter = ones(2) / (2*2);    % for mean    过滤器，滤波器（实现平均池化）
  image  = conv2(x(:, :, k), filter, 'valid');          %二维卷积函数conv2
                                                      %池化运算也算一种卷积运算
  y(:, :, k) = image(1:2:end, 1:2:end);
end

end
 