% Biffurcation diagram from 1/2e rule

function [x_pos, y_pos] = bif_diagram_BN_rule(deltas)
%     deltas = linspace(1,10,100);
    
    number_clusters = floor(deltas); 
    
    % number of symmetric clusters
    for i = 1:length(number_clusters)
        cluster = number_clusters(i);
        if rem(cluster,2) ==0
            number_sym_clusters(i) = cluster;
            number_central_clusters(i) = 0;
        else
            number_sym_clusters(i) = cluster-1;
            number_central_clusters(i) = 1;
        end
    end
    
    % positions of central clusters
    central_region = deltas - 1; % half of the central region
    
    plot(deltas(number_central_clusters>0), 0, 'b.')
    hold on
    
    x_pos = deltas(number_central_clusters>0);
    y_pos = zeros(size(x_pos));
    
    for j = 1:length(number_sym_clusters)
        k=0;
        if number_sym_clusters(j)>0
            for i = 1:number_sym_clusters(j)/2
                position = central_region(j) - 2*k;
%                 plot(deltas(j), position, 'b.');
                x_pos = [x_pos deltas(j)];
                y_pos = [y_pos position];
    
%                 plot(deltas(j), -position, 'b.');
%                 hold on
                k = k+1;
                x_pos = [x_pos deltas(j)];
                y_pos = [y_pos -position];
            end
        end
    end
%     hold off
end

% %% Simple dots of biffurcation
% number_clusters = deltas; 
% 
% % number of symmetric clusters
% for delta = deltas
%     if rem(delta,2) ==0
%         number_sym_clusters(delta) = delta;
%         number_central_clusters(delta) = 0;
%     else
%         number_sym_clusters(delta) = delta-1;
%         number_central_clusters(delta) = 1;
%     end
% end
% 
% % positions of clusters
% central_region = deltas - 1; % half of the central region
% 
% plot(deltas(number_central_clusters>0), 0, 'b.')
% hold on
% 
% for delta = deltas
%     k = 0;
%     for i = 1:number_sym_clusters(delta)
%         position = central_region(delta) - k;
%         plot(delta, position, 'b.');
%         plot(delta, -position, 'b.');
%         hold on
%         k = k+1;
%     end
% end
% hold off