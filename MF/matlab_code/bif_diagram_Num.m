function [x_pos_Num, y_pos_Num, bif_points] = bif_diagram_Num(filename)
load(filename)

 x_pos_Num = [];
 y_pos_Num = [];
for i = 1:max(NoStates)
    idx = find(Location(:,1,i));
    x_pos_Num = [x_pos_Num transpose(Value_delta(idx))];
    y_pos_Num = [y_pos_Num transpose(2*Value_delta(idx).*(Location(idx,1,i)-0.5))];
end

% Bifurcation points
for i=1:9
    clusters_indecies(i) = find(NoStates == i,1);
end
bif_points = Value_delta(clusters_indecies);
% plot(x_pos_Num, y_pos_Num, 'b.')
end
