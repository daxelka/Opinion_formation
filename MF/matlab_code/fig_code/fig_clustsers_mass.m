cd '/Users/daxelka/Research/Deffuant_model/Susan_code/fig_code'
addpath '/Users/daxelka/Research/Deffuant_model/Susan_code'

load('../central_cluster_mass.mat')
load('../symetric_cluster_mass.mat')

color1 = [0    0.4470    0.7410];
color6 = [0.3010    0.7450    0.9330];
light_blue = [114 147 203]./255;
gray_cyan =  [66, 198, 235]./255;

% l_blue = [153, 204, 255]./255;
% d_blue = [31, 80, 145]./255;

% l_blue = [143, 225, 255]./255;
l_blue = [153, 204, 255]./255;
% l_blue = [102, 153, 255]./255;
d_blue = [51, 51, 255]./255;

figure('Position', [0 0 550 500])
for i = 1:5
        p_central = plot(central_clusters_pos{i}, central_clusters_mass{i},'Color', l_blue, 'LineWidth', 2);
    hold on
end

for i = 1:4
%     p_symmetric = plot(symmetric_clusters_pos{i}, symmetric_clusters_mass{i}, 'Color', d_blue, 'LineWidth',2);
    p_symmetric = plot(symmetric_clusters_pos{i}, symmetric_clusters_mass{i}, 'Color', d_blue, 'LineWidth',1.5);
    hold on
end
% hold off

% Adding numerical bifurcation points
% Central clusters
bif_points = [2.725 4.93 7.11 9.28];
for i = 2:5
    bf_idx = find(central_clusters_pos{i} >= bif_points(i-1),1);
    x_bf = central_clusters_pos{i}(bf_idx);
    y_bf = central_clusters_mass{i}(bf_idx);
    scatter(x_bf, y_bf,  100, [0 0 0],'x','LineWidth',2);
    p_bif = xline(bif_points,'Color', [0.3 0.3 0.3], 'LineStyle', ':', 'LineWidth',1);
    hold on
end

% Symmetric clusters

bif_points = [1.8600, 4.0600, 6.2300, 8.3900];
for i = 1:4
    bf_idx = find(symmetric_clusters_pos{i} >= bif_points(i),1);
    x_bf = symmetric_clusters_pos{i}(bf_idx);
    y_bf = symmetric_clusters_mass{i}(bf_idx);
    scatter(x_bf, y_bf,  100, [0 0 0],'x','LineWidth',2);
    p_bif = xline(bif_points,'Color', [0.3 0.3 0.3], 'LineStyle', ':', 'LineWidth',1);
    hold on
end


hold off
box off

xlabel('\Delta')
ylabel('m/\epsilon')
set(gca, 'FontSize', 20)
% set(gca, 'YScale', 'log')
% ylim([4e-04, 1])

legend([p_central(1) p_symmetric(1)],'central cluster  ','symmetric cluster',...
        'Location', 'southoutside',...
        'Orientation','horizontal',...
        'Box','off',...
        'Fontsize',20, ...
        'NumColumns',2) 

%% Alternative version

% color1 = [0    0.4470    0.7410];
% color6 = [0.3010    0.7450    0.9330];
figure('Position', [0 0 550 500])
for i = 1:5
    p_central = plot(central_clusters_pos{i}, central_clusters_mass{i}, '--k', 'LineWidth', 1.2);
    hold on
end

for i = 1:4
    p_symmetric = plot(symmetric_clusters_pos{i}, symmetric_clusters_mass{i}, 'color', 'k', 'LineWidth', 1.2);
    hold on
end
% hold off

% Adding numerical bifurcation points
bif_points = [2.725 4.93 7.11 9.28];
for i = 2:5
    bf_idx = find(central_clusters_pos{i} >= bif_points(i-1),1);
    x_bf = central_clusters_pos{i}(bf_idx);
    y_bf = central_clusters_mass{i}(bf_idx);
%     p_bif = plot(x_bf, y_bf, 'k.', MarkerSize=6, Marker='+', LineWidth=1);
    p_bif = scatter(x_bf, y_bf,  60, 'b','o', 'filled','LineWidth',1.5);
%     p_bif = scatter(x_bf, y_bf,  120, 'b','s','LineWidth',1.5);
    hold on
end

bif_points = [1.8600, 4.0600, 6.2300, 8.3900];

y_bf_sym = [];
for i = 1:4
    bf_idx = find(symmetric_clusters_pos{i} >= bif_points(i),1);
    x_bf = symmetric_clusters_pos{i}(bf_idx);
    y_bf = symmetric_clusters_mass{i}(bf_idx);
%     p_bif = plot(x_bf, y_bf, 'k.', MarkerSize=6, Marker='+', LineWidth=1);
    p_bif = scatter(x_bf, y_bf,  60, 'b','o', 'filled','LineWidth',1.5);
%     p_bif = scatter(x_bf, y_bf, 120, 'b','s','LineWidth',1.5);
    y_bf_sym(i) = symmetric_clusters_mass{i}(bf_idx);
    hold on
end

% Adding LS bifurcation points
% bif_points_LS = load('bif_diagram_LS.mat', 'bifurcations_points');
% bif_points_LS = [1.7345, 2.7373, 3.8427, 4.9607, 6.0824, 7.2058, 8.3292, 9.4526];

% Central LS
% bif_points = [2.7373 4.9607 7.2058 9.4526];
% for i = 2:5
%     bf_idx = find(central_clusters_pos{i} >= bif_points(i-1),1);
%     x_bf = central_clusters_pos{i}(bf_idx);
%     y_bf = central_clusters_mass{i}(bf_idx);
% %     p_bif = plot(x_bf, y_bf, 'k.', MarkerSize=6, Marker='+', LineWidth=1);
% %     p_bif = scatter(x_bf, y_bf,  46, 'r','o', 'filled','LineWidth',1.5);
%     p_bif = scatter(x_bf, y_bf, 120, 'r','x','LineWidth',3);
%     hold on
% end

% Symmetric LS
% bif_points = [1.7345 3.8427 6.0824 8.3292];
% for i = 1:4
% %     bf_idx = find(symmetric_clusters_pos{i} >= bif_points(i),1);
% %     x_bf = symmetric_clusters_pos{i}(bf_idx);
%     x_bf = bif_points(i);
% %     y_bf = symmetric_clusters_mass{i}(bf_idx);
%     y_bf = y_bf_sym(i);
% %     p_bif = plot(x_bf, y_bf, 'k.', MarkerSize=6, Marker='+', LineWidth=1);
% %     p_bif = scatter(x_bf, y_bf,  46, 'r','o', 'filled','LineWidth',1.5);
%     p_bif = scatter(x_bf, y_bf, 120, 'r','x','LineWidth',3);
%     hold on
% end


hold off
box off

xlabel('\Delta')
ylabel('m/\epsilon')
set(gca, 'FontSize', 20)
% set(gca, 'YScale', 'log')
% ylim([4e-04, 1])

legend([p_central(1) p_symmetric(1)],'central cluster  ','symmetric cluster',...
        'Location', 'southoutside',...
        'Orientation','horizontal',...
        'Box','off',...
        'Fontsize',20, ...
        'NumColumns',2) 

% legend([p_central(1) p_symmetric(1) p_bif(1) p_bif(1)],'central cluster','symmetric cluster', 'LS bif. points','Numeric bif. points',...
%         'Location', 'southoutside',...
%         'Orientation','horizontal',...
%         'Box','off',...
%         'Fontsize',20, ...
%         'NumColumns',2) 

