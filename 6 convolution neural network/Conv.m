function y = Conv(x, W)
%Conv函数接受输入图像和卷积过滤器矩阵，返回特征映射
%

[wrow, wcol, numFilters] = size(W);
[xrow, xcol, ~         ] = size(x);

yrow = xrow - wrow + 1;              %卷积后的大小
ycol = xcol - wcol + 1;

y = zeros(yrow, ycol, numFilters);

for k = 1:numFilters
  filter = W(:, :, k); 
  filter = rot90(squeeze(filter), 2);         %逆方向旋转矩阵（图片）2*90度
  y(:, :, k) = conv2(x, filter, 'valid');     %conv2为MATLAB内置的卷积函数
end

end

