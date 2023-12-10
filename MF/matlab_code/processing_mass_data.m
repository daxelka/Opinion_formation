%% Load data
load('Pfinal_delta.mat')

% Plot of cluster locations vs delta
max_mass_value = max(Mass(:));
% colors
C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]};

% bifurcation numeric
for i=1:9
    clusters_indecies(i) = find(NoStates == i,1);
end
bifurcations_numeric = Value_delta(clusters_indecies);

central_cluster_intervals = [[0.5 1.86]; [2.445 4.06]; [4.66, 6.23]; [6.84, 8.39]; [9.005, 10]];

% central_clusters_pos = [];
% central_clusters_mass = [];
% symmetric_clusters_pos = [];
% symmetric_clusters_mass = [];
% cluster_number = [];

for i = 1:5
    idx = find(Location(:,1,i));
    x_pos = Value_delta(idx);
    y_pos = (Mass(idx,1,i))/max_mass_value;
    interval = central_cluster_intervals(i,:);
    sampling = (x_pos >= interval(1)) & (x_pos <= interval(2));

    central_clusters_pos{i} = x_pos(sampling);
    central_clusters_mass{i} = y_pos(sampling);
    symmetric_clusters_pos{i} = x_pos(~sampling);
    symmetric_clusters_mass{i} = y_pos(~sampling);
    cluster_number{i} = i * ones(size(x_pos(sampling)));

    plot(x_pos(sampling),y_pos(sampling),'r.');
    hold on
    plot(x_pos(~sampling),y_pos(~sampling),'b.');
    hold on
end