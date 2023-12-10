% deltas = linspace(1,10,500);
% cd '/Users/daxelka/Research/Deffuant_model/Susan_code/fig_code'
% addpath '/Users/daxelka/Research/Deffuant_model/Susan_code'

% colors
% l_blue = [102, 153, 255]./255;
l_blue = [153, 204, 255]./255;
d_blue = [51, 51, 255]./255;

% Numerical diagram

filename = 'Pfinal_delta.mat';

[x_pos_Num, y_pos_Num, bif_points_Num] = bif_diagram_Num(filename);

figure('Position', [0 0 550 500])
p_Num = plot(x_pos_Num, y_pos_Num, 'Color', 'b','LineStyle', 'none', 'Marker','.');
hold on

% Plotting corrections to bifurcation points of the central cluster
% bifurcations_numeric = [0.5000,  1.8600, 2.4450, 4.0600,   4.6600, 6.2300, 6.8400, 8.3900,  9.0050]; 
bifurcations_numeric = bif_points_Num;
biffurcations_corrected = [1.8600, 2.725, 4.0600, 4.93, 6.2300, 7.11, 8.3900, 9.28];
for i = 2:length(biffurcations_corrected)
    line([bifurcations_numeric(i+1)-0.05 biffurcations_corrected(i)],[0 0],'Color','white','LineWidth',5)
    hold on
end

% Linear Stability diagram
load('bif_diagram_LS.mat')

for n_line = 1:1:10
%     p_LS = plot(delta_coor{n_line}, peak_position_delta{n_line}, 'r--', LineWidth=2);
    p_LS = plot(delta_coor{n_line}, peak_position_delta{n_line}, 'r--', 'LineWidth', 2);
    hold on
end

% Invisible blue line just for the legend
% p_blue_line = plot([1, 1.2], [0, 0], 'b', LineWidth=2);
p_blue_line = plot([1, 1.2], [0, 0], 'b', 'LineWidth', 2);
hold on

hold off
box off
% xticks([0 1 2 3 4 5 6 7 8 9 10])
xlabel('\Delta')
ylabel('y')
set(gca,'Fontsize', 20)
legend([p_LS(1) p_blue_line(1)],'Linear stability ','Numerical solution',...
        'Location', 'southoutside',...
        'Orientation','horizontal',...
        'Box','off',...
        'Fontsize',20) 

% % 1/2e rule diagram
% [x_pos_BN, y_pos_BN] = bif_diagram_BN_rule(deltas);
% plot(x_pos_BN, y_pos_BN, 'g.')
% hold on
% bif_points_BN = 2:1:9;
% % xline(bif_points_BN, 'k:')
% for i = bif_points_BN
%     line([i i], [-1, 1], 'LineStyle', ':', 'LineWidth', 1)
%     hold on
% end
% hold on

