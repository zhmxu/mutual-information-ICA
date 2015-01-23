function [Wica_best, Wica_all_initializations, negentropy_all_initializations, negentropy_vs_rotation_all_initializations] = minimize_mutual_information_via_rotation(Wpca, n_random_initializations, random_seed, plot_figures)

if nargin < 3
    random_seed = 1;
end

if nargin < 4
    plot_figures = 1;
end

% gaussian entropy
gaussEntropy = log(sqrt(2*pi*exp(1)));

% maximum allowed number of iterations
max_iter = 1000;

% how finely to sample possible rotations
resolution = 61;

% resolution must be odd
if mod(resolution,2) == 0
    resolution = resolution+1;
end

% reinitializing random seed
sd = RandStream('mt19937ar','Seed', random_seed);
try 
    RandStream.setDefaultStream(sd); %#ok<SETRS>
catch
    RandStream.setGlobalStream(sd);
end

% initialization
n = size(Wpca,1);
pairs = flipud(combnk(1:n,2));
n_pairs = size(pairs,1);

% rotation search grid
th = linspace(-pi/4,pi/4, resolution);

% summed negentropy for all random initializations
% the function maximizes this quantity
negentropy_all_initializations = nan(1,n_random_initializations);

% ICA weights for all runs of the algorithm with different random
% initializations
Wica_all_initializations = nan([size(Wpca),n_random_initializations]);

% entropy between pairs of components for different rotations
negentropy_vs_rotation_all_initializations = nan(length(th),n_pairs,n_random_initializations);

% run optimization
for z = 1:n_random_initializations
    
    % randomly rotate PCA dimensions
    rotMat = random_rotation_matrix(n);
    Wica_all_initializations(:,:,z) = rotMat*Wpca;
   
    % negentropy of projection
    negentropy = nan(1,max_iter+1);
    x = nan(1,n);
    for j = 1:n
        x(j) = gaussEntropy - entropy(Wpca(j,:));
    end
    negentropy(1) = mean(x);
    
    q = 0;
    while 1
        
        % rotate pairs
        q = q+1;
        rot = nan(1,n_pairs);
        negentropy_vs_rotation_all_initializations(:,:,z) = nan(length(th),n_pairs);
        
        % loop through each pair in random order
        for i = randperm(n_pairs)
            % compute entropy for all rotations
            for j = 1:length(th)
                rotMat = [cos(th(j)), -sin(th(j)); sin(th(j)), cos(th(j))];
                Vrot = rotMat*Wica_all_initializations(pairs(i,:),:,z);
                x = gaussEntropy - [entropy(Vrot(1,:)), entropy(Vrot(2,:))];
                negentropy_vs_rotation_all_initializations(j,i,z) = mean(x);
            end
            [~,rot(i)] = max(negentropy_vs_rotation_all_initializations(:,i,z));
            rotMat = [cos(th(rot(i))), -sin(th(rot(i))); sin(th(rot(i))), cos(th(rot(i)))];
            Wica_all_initializations(pairs(i,:),:,z) = rotMat*Wica_all_initializations(pairs(i,:),:,z);
        end
        
        % compute mean negentropy
        x = nan(1,n);
        for j = 1:n
            x(j) = gaussEntropy - entropy(Wica_all_initializations(j,:,z));
        end
        negentropy(q+1) = mean(x);
        if plot_figures
            figure(1); clf(1);
            plot(th/pi,negentropy_vs_rotation_all_initializations(:,:,z));
            title(sprintf('Rep: %d, Iteration: %d, NegEnt: %.4f', z, q, negentropy(q+1)));
            xlabel('Pi Radians'); ylabel('Mean NegEntropy');
            drawnow;
        end
        
        % quit out if data were not rotated
        if all(rot == (resolution+1)/2)
            break;
        end
    end
    
    % store negentropy for this run
    negentropy_all_initializations(z) = negentropy(q+1);
end

% select best run
[~,xi] = max(negentropy_all_initializations);
Wica_best = Wica_all_initializations(:,:,xi);
negentropy_vs_rotation_best_initialization = negentropy_vs_rotation_all_initializations(:,:,xi); 
negentropy_best = negentropy_all_initializations(xi);

if plot_figures
    figure(1); clf(1);
    plot(th/pi,negentropy_vs_rotation_best_initialization);
    title(sprintf('Best run, NegEnt: %.4f', negentropy_best));
    xlabel('Pi Radians'); ylabel('Mean NegEntropy');
    
    figure;
    plot(negentropy_all_initializations,'k-o','LineWidth',2);
    xlim([0 n_random_initializations+1]);
    ylabel('Mean NegEntropy'); xlabel('Runs');
    title('NegEntropy for Each Run');
end

