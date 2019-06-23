function y = Sigmoid(x)
  y = 1 ./ (1 + exp(-x));  %  ./代表对向量各元素执行除法，结果仍为向量
end