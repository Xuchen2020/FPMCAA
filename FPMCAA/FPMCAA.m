function [Y, obj, de, alpha] = FPMCAA(X, Y, numanchor)

nv = length(X);
Z = cell(1, nv); 
Av = cell(1, nv);
k = length(unique(Y));

m = numanchor;
n = size(X{1}, 2);
lambda = 0.1;
if n == 2500  % || n==38654
    m = 2 * numanchor;
end

Q = cell(1,nv);
Y = eye(k,n); 

for kk = 1:nv
    d_v = size(X{kk}, 1);
    Av{kk} = zeros(d_v, m);
    Z{kk} = zeros(m, n);
    Q{kk} = eye(m,k);
end

pp = zeros(m,n);
alpha = ones(1,nv)/nv;
obj = zeros(1,20);
de = obj;
prevObjective = 0;

for iter = 1:100
    %%   A 
    for ia = 1:nv
        C = alpha(ia)^2 * X{ia}*Z{ia}';      
        [U,~,V] = svd(C,'econ');
        Av{ia} = U*V';
    end

    %% Z 
    % Z_sum = zeros(m,n);
    Zq = zeros(k,n);
    P = zeros(nv,1);
    % for v = 1:nv 
        for a=1:nv
            M = (alpha(a)^2*Av{a}'*X{a}+ lambda * Q {a} * Y)/(alpha(a)^2 + 1 + lambda); 
            idx = 1:m;
            for ii=1:n
                pp(idx,ii) = EProjSimplex_new(M (idx,ii));           
            end
            Z{a} = pp;
            clear pp;
            Zq = Zq + Q{a}'*Z{a};
        end
    
    %% optimize alpha
 
    for iv = 1:nv
        P(iv) = norm( X{iv} - Av{iv} * Z{iv},'fro')^2;
    end
    Mfra = P.^-1;
    Q_1 = 1/sum(Mfra);
    alpha = Q_1*Mfra;
 
    %% updata Y
   [Uy, ~, Vy] = svd(Zq,'econ');   
    Y = Uy*Vy';
   clear Uy Vy
    %% updata Q
    objective = 0;
    for idx = 1:nv     
        if n == 38654
            [Uq,~,Vq] = svd(Z{idx}*Y','econ');
            Q{idx} = Uq*Vq'; 
        else
            Q{idx} = Z{idx}*Y';  
        end
        term1 = alpha(idx)^2 * P(idx);
        term2 = lambda * norm(Z{idx} - Q{idx} * Y, 'fro')^2;
        term3 = norm(Z{idx}, 'fro')^2;
        objective = objective + term1 + term2 + term3;
    end
    % fprintf('objective %7.10f \n', objective); 

    obj(iter) = objective;
    de(iter) = abs(prevObjective - objective)/prevObjective; 
    prevObjective = objective;

    if (iter >= 2 && ( de(iter) <1e-5 || objective < 1e-2 )) 
            break;
    end
end
end