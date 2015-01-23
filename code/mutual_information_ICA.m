function [Rica, Wica, Rpca, Wpca] = mutual_information_ICA(X, K, n_random_initializations, random_seed, plot_figures)
% function mutual_information_ICA(X, K, n_repetitions, plot_figures, random_seed)
% 
% Top-level script for running mutual-information based ICA analysis. X is
% an [M x N] input matrix, with N data points, each with M features. 
% The input matrix is factored into a [M x K] matrix (Rica) with K "basis
% functions" in its columns, and a [K x N] matrix of weights (Wica) specifying the
% contribution of each discovered basis function to each data point. 
% 
% The cost function optimized is non-convex and is thus sensitive to local minima.
% To address this issue, the algorithm is run multiple times with different
% random initialization and the answer with the lowest cost is returned. 
% n_random_initializations specifies the number of random initializations (default is 10).
% 
% Before any runs of the algorithm, the random stream is reset with the 
% seed specified by "random_seed" (default is 1). The solutions returned
% by the algorithm should be the same as long as the seed is the same.
% 
% The function also returns the top K PCA components in Rpca and Wpca. 
% 
% The algorithm iteratively rotates the top K principal components to
% maximize the "negentropy" between components. This is equivalent to
% minimizing the mutual information between components, given the
% constraint that component's weights are uncorrelated. 
% 
% Because negentropy is estimated with a histogram, the algorithm tends to
% work well with a large number of data points (~10,000). The run-time of
% the algorihm will scale with nchoosek(K,2) where K is the number of
% components. 
% 
% See Hyvärinen and Oja, 2000. Independent Component Analysis:
% Algorithms and Applications. http://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf.
% 
% Entropy is estimated using a script from Rudy Moddemeijer.
% See http://www.cs.rug.nl/~rudy/matlab/
% 
% Example:
% M = 100;
% N = 10000;
% K = 3;
% R = rand(M,K);
% W = gamrnd(1,1,[K,N]);
% X = R*W + 0.1*randn(M,N);
% n_random_initializations = 10;
% random_seed = 1;
% plot_figures = 0;
% [Rica, Wica, Rpca, Wpca] = mutual_information_ICA(X, K, n_random_initializations, random_seed, plot_figures);
% corr(R,Rpca)
% corr(R,Rica)

if nargin < 3
    n_random_initializations = 3;
end

if nargin < 4
    plot_figures = 0;
end

if nargin < 5
    random_seed = 1;
end

% matrix dimensions
[M,N] = size(X);

% demean rows
X_zero_mean_rows = nan(size(X));
for i = 1:M
    X_zero_mean_rows(i,:) = X(i,:) - mean(X(i,:));
end

% top K PCA components
[U,S,V] = svd(X_zero_mean_rows,'econ');
Rpca = U(:,1:K) * S(1:K,1:K);
Wpca = V(:,1:K)';

% rotate PCA components to minimize mutual information
Wica = minimize_mutual_information_via_rotation(Wpca, n_random_initializations, random_seed, plot_figures);

% estimate basis functions
Rica = X_zero_mean_rows*pinv(Wica);

% normalize to have unit RMS
for i = 1:K
    Rica(:,i) = Rica(:,i)/sqrt(mean(Rica(:,i).^2));
end

% re-estimate weights with non-demeaned data
Wica = pinv(Rica)*X;

% orient so that average weights are positive
Rica = Rica .* repmat(sign(mean(Wica,2))',M,1);
Wica = Wica .* repmat(sign(mean(Wica,2)),1,N);