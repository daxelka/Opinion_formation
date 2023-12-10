%% Save location of cosine peaks
deltas = linspace(1,10,1000);
eps = 1./deltas/2;
num_modes = n_mode(eps);
eps = eps(num_modes > 0);
num_modes = num_modes(num_modes > 0);

peak_locations = zeros(0,3);
for nn = sort(unique(num_modes))
    eps_vals = eps(num_modes == nn);
    for mm = 1:ceil((nn+1)/2)
        ll = zeros(length(eps_vals),3);
        ll(:,2) = eps_vals;
        ll(:,1) = eps_vals + (mm-1)*(1-2*eps_vals)/nn;
        ll(:,3) = mm;
        peak_locations = [ll ; peak_locations];
    end
end

ll = peak_locations;
ll(:,1) = 1-ll(:,1);
ll(:,3) = ll(:,3) + 5;
peak_locations = [peak_locations ; ll];

% bifurcation analytic
for i=1:8
    clusters_indecies(i) = find(num_modes == i,1);
end
bifurcations_analytic = 1./eps(clusters_indecies)/2

% Plot peaks
x_eps = peak_locations(:,2);
y_eps = peak_locations(:,1);
x_delta = 1./x_eps/2;
y_delta = (2 .* y_eps-1)./x_eps/2;
plot(x_delta, y_delta, 'r.')


%%  Plotting analytic results as connected lines
for n_line = 1:1:10
    sampling_n_line = peak_locations(:,3) == n_line;
    x_temp = peak_locations(sampling_n_line,2);
    y_temp = peak_locations(sampling_n_line,1);
    x_delta_temp = 1./x_temp/2;
    y_delta_temp = (2 .* y_temp-1)./x_temp/2;
    data_temp = [x_delta_temp, y_delta_temp];
    data_sorted = sortrows(data_temp,1);
    plot(data_sorted(:,1), data_sorted(:,2), 'r')
    hold on
end
hold off

%% Plot of cluster locations vs delta
% This is what we need, plots biffurcation diagramm
load('Pfinal_delta.mat')

% bifurcation numeric
for i=1:9
    clusters_indecies(i) = find(NoStates == i,1);
end
bifurcations_numeric = Value_delta(clusters_indecies)

for i = 1:max(NoStates)
    idx = find(Location(:,1,i));
    plot(Value_delta(idx),2*Value_delta(idx).*(Location(idx,1,i)-0.5),'b.')
    hold on
end
xticks([0 1 2 3 4 5 6 7 8 9 10])
% plot(x_delta, y_delta, 'r.')

% Add analytic results
for n_clusters = 1:1:10
    sampling_n_clusters = peak_locations(:,3) == n_clusters;
    x_temp = peak_locations(sampling_n_clusters,2);
    y_temp = peak_locations(sampling_n_clusters,1);
    x_delta_temp = 1./x_temp/2;
    y_delta_temp = (2 .* y_temp-1)./x_temp/2;
    data_temp = [x_delta_temp, y_delta_temp];
    data_sorted = sortrows(data_temp,1);
    plot(data_sorted(:,1), data_sorted(:,2), 'r-', 'LineWidth',2)
    hold on
end
% hold off

% Plotting corrections to bifurcation points of the central cluster
biffurcations_corrected = [1.8600, 2.725, 4.0600, 4.93, 6.2300, 7.11, 8.3900, 9.28];
for i = 2:length(biffurcations_corrected)
    line([bifurcations_numeric(i+1)-0.05 biffurcations_corrected(i)],[0 0],'Color','white','LineWidth',5)
    hold on
end
xlabel('\Delta')
ylabel('y')
set(gca,'Fontsize', 18)
% hold on
% hold off

% Ploting diagram by BN rule

% n_clusters_BN = floor(deltas);
% xline(n_clusters_BN)
% hold off

% [x_pos_BN, y_pos_BN] = bif_diagram_BN_rule(deltas);
% plot(x_pos_BN, y_pos_BN, 'g.')

bif_points_BN = 2:1:9;
% xline(bif_points_BN, 'k:')
for i = bif_points_BN
    line([i i], [-2, 2], 'LineStyle', '-.', 'LineWidth', 1, 'color', 'k')
    hold on
end
hold off

%% Functions
function[s_val] = s_over_2eps(x)
    s_val = zeros(size(x));
    x_temp = x(x~=0);
    s_val(x~=0) = (1./x_temp).*(4.*sin(x_temp./2)-sin(x_temp))-1;
end

function[n] = n_mode(eps)
    %function to calculate the fastest growing mode for a given eps
    %calculate x_max
    syms x
    f = ((4*sin(x/2)-sin(x))./x)-1;
    x_max = vpasolve(diff(f,x)==0,x,[0.01,5]); %need lower boundary of interval to be >0 as s has a local min at x=0. Upper boundary: s<0 for x>5.
    n_max = (1-2.*eps).*x_max./(eps.*2.*pi); % noninteger value of n correponding to x = 2.8 (max of s/2eps)
    x_below = 2.*floor(n_max).*pi.*eps./(1-2.*eps); %value of x corresponding to the closest integer value of n below n_max
    x_above = 2.*ceil(n_max).*pi.*eps./(1-2.*eps); %value of x corresponding to the closest integer value of n above n_max

    n_ind = s_over_2eps(x_below)> s_over_2eps(x_above); %cases where n below n_max has a bigger value of s than n above n_max
    n(n_ind) = floor(n_max(n_ind));
    n(~n_ind) = ceil(n_max(~n_ind));
    n = double(n);
end