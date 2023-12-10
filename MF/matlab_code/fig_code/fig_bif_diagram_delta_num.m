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

% Corrections to numerical diagram
biffurcations_corrected = [1.8600, 2.725, 4.0600, 4.93, 6.2300, 7.11, 8.3900, 9.28];
for i = 2:length(biffurcations_corrected)
    line([bif_points_Num(i+1)-0.05 biffurcations_corrected(i)],[0 0],'Color','white','LineWidth',5)
    hold on
end
hold off


box off
% xticks([0 1 2 3 4 5 6 7 8 9 10])
xlabel('\Delta')
ylabel('y')
% xlim([0 1])
set(gca,'Fontsize', 28)

