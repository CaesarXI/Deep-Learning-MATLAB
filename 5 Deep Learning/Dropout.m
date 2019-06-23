function ym = Dropout(y, ratio)
  [m, n] = size(y);  
  ym     = zeros(m, n);
  
  num     = round(m*n*(1-ratio));      %round函数：四舍五入   num为保留元素个数
  idx     = randperm(m*n, num);        %取出保留元素的索引
  ym(idx) = 1 / (1-ratio);             %赋值给保留元素
end