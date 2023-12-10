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

x_tilda = 1./x_pos_Num/2;
y_tilda = y_pos_Num ./ x_pos_Num /4 + 1/2;

figure('Position', [0 0 550 500])
p_Num = plot(x_tilda, y_tilda, 'Color', 'b','LineStyle', 'none', 'Marker','.');


box off
% xticks([0 1 2 3 4 5 6 7 8 9 10])
xlabel('\epsilon')
ylabel('x')
xlim([0.03 0.5])
set(gca,'Fontsize', 28)

