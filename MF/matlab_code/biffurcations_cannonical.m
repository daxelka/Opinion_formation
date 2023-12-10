load('Pfinal_delta.mat')
i=1;
 idx = find(Location(:,1,i));
    delta_temp = Value_delta(idx);
    locations_temp = 2*Value_delta(idx).*(Location(idx,1,i)-0.5);
%     locations_num = [locations_temp(locations_temp<=0)+delta_temp(locations_temp<=0)];
%     delta_num = [delta_temp(locations_temp<=0)];
    locations_num = [locations_temp+delta_temp];
    delta_num = [delta_temp];

for i = 2:max(NoStates)
    idx = find(Location(:,1,i));
    delta_temp = Value_delta(idx);
    locations_temp = 2*Value_delta(idx).*(Location(idx,1,i)-0.5);
%     locations_num = [locations_num; locations_temp(locations_temp<=0)+delta_temp(locations_temp<=0)];
%     delta_num = [delta_num; delta_temp(locations_temp<=0)];
    locations_num = [locations_num; locations_temp+delta_temp];
    delta_num = [delta_num; delta_temp];
end

% plot(x_tilde_num,locations_num,'b.')
hist(locations_num, 200, 'Normalization','pdf')
xlabel('$\tilde{x}=x/\epsilon$','Interpreter','latex')
xticks([0 5 10])
ylabel('$\tilde{P}(\tau=\infty)$','Interpreter','latex')
set(gca, 'Fontsize', 20)
% plottools('on')
% for i = 1:max(NoStates)
%     idx = find(Location(:,1,i));
%     x_tilde_temp = Value_delta(idx);
%     locations_temp = 2*Value_delta(idx).*(Location(idx,1,i)-0.5);
%     locations_num = locations_temp(locations_temp<=0);
%     x_tilde_num = x_tilde_temp(locations_temp<=0);
% 
%     plot(x_tilde_num,locations_num,'b.')
%     hold on
% end
% hold off

% colors = 'bgrky';
% k=1;
% for i = 1:4
%     idx = find(Location(:,1,i));
%     size(Value_delta(idx));
%     deltas_num(:,i) = Value_delta(idx);
%     locations_temp = 2*Value_delta(idx).*(Location(idx,1,i)-0.5);
%     locations_num(:,i) = locations_temp(locations_temp<=0);
%     k = k+1;
%     scatter(Value_delta(idx),2*Value_delta(idx).*(Location(idx,1,i)-0.5), colors(i))
%     hold on
% end

%% 
% Save location of cosine peaks
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
% plot(x_delta, y_delta, 'r.')

%% Plotting analytic results as connected lines

    sampling_n_line = peak_locations(:,3) == 1;
    y_temp = peak_locations(sampling_n_line,1)./peak_locations(sampling_n_line,2);
    data_temp = [y_temp];

for n_line = 2:1:10
    sampling_n_line = peak_locations(:,3) == n_line;
    y_temp = peak_locations(sampling_n_line,1)./peak_locations(sampling_n_line,2);
    data_temp = [data_temp; y_temp];
end
hist(data_temp, 1000, 'Normalization','pdf')

%%%% Functions
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