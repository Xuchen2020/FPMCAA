clc;
clear;

addpath /home/xu.chen/code/1-mvsc/
addpath E:\学习\multiview\code\1-mvsc\Tmvsc
addpath E:\学习\multiview\code\1-mvsc\ClusteringMeasure
addpath E:\学习\multiview\code\dataset
addpath(genpath('manopt'))
addpath ClusteringMeasure

Dname = 'BDGP_mv.mat';
load(Dname);
Rname = ['Res_time_' Dname];
disp(Rname)
Y = double(Y)+1;
gnd = Y;
nv = length(X);
n = length(gnd);

numClust = length(unique(gnd));

for nv_idx = 1 : nv
     X{nv_idx} = normc(X{nv_idx}');
end



anchor_number = [numClust, 2*numClust, 3*numClust, 4*numClust, 5*numClust, 6*numClust, 7*numClust];



for anchor_index = 1 %length(anchor_number)

    tic
    [F, obj, de, alpha] = FPMCAA(X, Y, anchor_number(anchor_index));
    [max_kmeans, mean_kmeans, std_kmeans] = result(F', gnd, numClust);
    timer = toc;

end

save(Rname,"mean_kmeans","std_kmeans","timer")
