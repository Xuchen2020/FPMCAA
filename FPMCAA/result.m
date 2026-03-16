function [resmax, res_mean, res_std]= result(Y,gnd,numclass)

stream = RandStream.getGlobalStream;
reset(stream);
U_normalized = Y ./ repmat(sqrt(sum(Y.^2, 2)), 1,size(Y,2));
maxIter = 10;

for iter = 1:maxIter
    y = kmeans(U_normalized,numclass,'maxiter',1000,'replicates',20,'EmptyAction','singleton');
    result(iter,:) = Clustering8Measure(gnd,y);
end
resmax = max(result,[],1);
res_mean = mean(result,1);
res_std = std(result,1);
