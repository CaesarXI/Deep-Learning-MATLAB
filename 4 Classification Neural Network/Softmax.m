function y = Softmax(x)
  ex = exp(x);            % x为向量，ex仍为向量；
  y  = ex / sum(ex);        %sum直接对向量求和
end