function y = Softmax(x)
  ex = exp(x);            % xΪ������ex��Ϊ������
  y  = ex / sum(ex);        %sumֱ�Ӷ��������
end