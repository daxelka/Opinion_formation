
peak_canonical = [1.26,3.44,5.61,7.74,9.89];

deltas = linspace(1,10,500);

% Simple version

% for delta = deltas
%     % cut the domain 
%     domain_lenght = delta;
%     % pinpoint peaks
%     peaks_in_domain = peak_canonical(peak_canonical<=domain_lenght-1);
%     % transform to local delta grid
%     peaks_local = peaks_in_domain - delta;
%     
%     scatter(delta*ones(size(peaks_local)), peaks_local, [],'b', 'filled');
%     hold on
%     scatter(delta*ones(size(peaks_local)), -peaks_local, [],'b', 'filled');
%     hold on
% end
% hold off

% plot parameters

l_blue = [0, 153, 204]./255;
d_blue = [51, 51, 255]./255;
gray = [0.75, 0.75, 0.75];

marker_size = 12;
marker_size_small = 8;

figure('Position', [0 0 550 500])

% Add the numerical diagram

filename = 'Pfinal_delta.mat';

[x_pos_Num, y_pos_Num, bif_points_Num] = bif_diagram_Num(filename);

p_Num = plot(x_pos_Num, y_pos_Num, 'Color', 'b','LineStyle', 'none', 'Marker','.', 'MarkerSize',8);
hold on

% Corrections to numerical diagram
biffurcations_corrected = [1.8600, 2.725, 4.0600, 4.93, 6.2300, 7.11, 8.3900, 9.28];
for i = 2:length(biffurcations_corrected)
    line([bif_points_Num(i+1)-0.05 biffurcations_corrected(i)],[0 0],'Color','white','LineWidth',5)
    hold on
end

% Version with central clusters
peaks_central_all = [];
deltas_central_all = [];
delta_central_true = [];
for delta = deltas
    % cut the domain 
    domain_lenght = delta;

    % pinpoint symmetric peaks
    peaks_in_domain = peak_canonical(peak_canonical<=domain_lenght-0.5);
    % transform to local delta grid
    peaks_local = peaks_in_domain - delta;
   
%     scatter(delta*ones(size(peaks_local)), peaks_local, marker_size, d_blue, 'filled');
    plot(delta*ones(size(peaks_local)), peaks_local, 'r.');
    hold on
%     scatter(delta*ones(size(peaks_local)), -peaks_local, marker_size, d_blue, 'filled');
    plot(delta*ones(size(peaks_local)), -peaks_local, 'r.');
    hold on

    % pinpoint central peaks
    peaks_in_centre = peak_canonical((peak_canonical>domain_lenght-0.5)&(peak_canonical<domain_lenght));
    peaks_central_local = peaks_in_centre -delta;
    deltas_central = delta*ones(size(peaks_in_centre));

    peaks_central_all = [peaks_central_all peaks_central_local];
    deltas_central_all = [deltas_central_all deltas_central];

%     scatter(delta*ones(size(peaks_central_local)), peaks_central_local, marker_size_small, gray, '.');
%     hold on
%     scatter(delta*ones(size(peaks_central_local)), -peaks_central_local, marker_size_small, gray, '.');
%     hold on
%     
    % move central peaks to a centre
    if ~isempty(peaks_in_centre)
%         scatter(delta, 0, marker_size, l_blue,'filled');
%         plot(delta, 0, 'Color', l_blue,'LineStyle','none', 'Marker','.');
%         hold on
        delta_central_true = [delta_central_true delta];
    end
end
% scatter(deltas_central_all(1:10:end), peaks_central_all(1:10:end),marker_size, d_blue, 'filled');
plot(deltas_central_all(1:10:end), peaks_central_all(1:10:end),'r.');
hold on
% scatter(deltas_central_all(1:10:end), -peaks_central_all(1:10:end),marker_size, d_blue, 'filled');
plot(deltas_central_all(1:10:end), -peaks_central_all(1:10:end),'r.');
hold on

p2 = plot(delta_central_true, zeros(size(delta_central_true)), 'Color', 'r','LineStyle','none', 'Marker','.');

hold on
% hold off

xlabel('\Delta')
ylabel('y')
set(gca,'Fontsize', 20)
legend([p2(1) p_Num(1)],'from canon. solution ','numerical solution',...
        'Location', 'southoutside',...
        'Orientation','horizontal',...
        'Box','off',...
        'Fontsize',20) 
hold off
