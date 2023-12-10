function [x_eps, y_eps, n_line, bif_points] = bif_diagram_LS(deltas)
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

% Spliting into separate lines and sorting elements
for n_line = 1:1:10
    sampling_n_line = peak_locations(:,3) == n_line;
    x_eps = peak_locations(sampling_n_line,2);
    y_eps = peak_locations(sampling_n_line,1);
    x_delta_temp = 1./x_eps/2;
    y_delta_temp = (2 .* y_eps-1)./x_eps/2;
    % Sorting data in delta
    data_sorted = sortrows([x_delta_temp, y_delta_temp],1);
    delta_coor{n_line}= data_sorted(:,1);
    peak_position_delta{n_line}= data_sorted(:,2);
end

% Biffurcation poitns
for i=1:8
    clusters_indecies(i) = find(num_modes == i,1);
end
bif_points = 1./eps(clusters_indecies)/2;

% How to plot the data
% for n_line = 1:1:10
%     plot(delta_coor{n_line}, peak_position_delta{n_line}, 'r.')
%     hold on
% end
% hold off

% Functions
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
end