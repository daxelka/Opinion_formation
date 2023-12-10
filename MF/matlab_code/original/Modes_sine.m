%% Plot of s as a function of x = eps*k
%figure
x = linspace(0,5);
y = s_over_2eps(x);
plot(x,y)
ylim([-0.4 0.4])
ylabel('$$\tilde{s}$$', 'Interpreter', 'LaTeX')
xlabel('$$\tilde{k}$$', 'Interpreter', 'LaTeX')

% largest value of x is approx 2.8
%% Save figures
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 8.6 7]);
saveas(gcf,'../../Thesis/4_Deffuant_MF_analysis/figs/s_vs_x.eps','epsc');
%saveas(gcf,'../Thesis_chapter/figs/s_vs_x.eps','epsc');

%% Plot sin(x)/x
a = 1/10;
x = linspace(0,100,1000);
y = sin(a*x)./(a*x);
plot(x,y)

hold on
y =  sin((1-a).*x)./((1-a).*x);
%plot(x,y)

y = sin(a*x)./(a*x) - sin(x)./x;
plot(x,y)

y =  sin(x)./x;
%plot(x,y)
plot(x,0*x)
hold off

%% Calculate max of s
syms x
f = ((4*sin(x/2)-sin(x))./x)-1;
x_max = vpasolve(diff(f,x)==0,x,[0.01,5]); %need lower boundary of interval to be >0 as s has a local min at x=0
x_0 =  vpasolve(f==0,x,[0.01,5]);

%% Plot of the derivative of s
g = matlabFunction(diff(f,x));
xvals = linspace(-1,5);
yvals = feval(g,xvals);
plot(xvals,yvals)


%% Fastest growing modes
eps = linspace(0.05,0.5);
n = n_mode(eps);
plot(eps,n)
ylabel('n')
xlabel('\epsilon')

%% Save location of cosine peaks
eps = linspace(0.05,0.5,1000);
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

%% Plot peaks
plot(peak_locations(:,2), peak_locations(:,1), '.')

%% Save peak_locations
filename = "../Data/peak_locations_dbmf";
save(filename, 'peak_locations')


%% Save location of cosine peaks DBMF
eps = linspace(0.05,0.5,1000);
mu = 0.5;
kk = 5;
qq = 0.9;

degrees = [kk,100];
degree_dist = [qq,1-qq];

q = degree_dist;
edge_probs = degrees'*degrees; %Not normalised by N<k>
coeffs = edge_probs.*repmat(q,length(q),1);
coeffs = coeffs./(sum(q*coeffs));

num_modes = n_mode_lambda(eps,mu,coeffs);

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

%% Plot cosine peaks - epsilon
figure
eps = linspace(0.05,0.5,1000);
n = n_mode(eps);

hold on
for i = 1:length(eps)
    m = 0:n(i);
    x = eps(i)+m.*(1-2.*eps(i))./n(i);
    plot(eps(i)*ones(1,length(m)),x,'.r')
end
title('Location of peaks for fastest growing modes')
xlabel('\epsilon')
ylabel('x')
hold off





%%
for ii = unique(n)
    for jj = 1:ii
        x = eps(n==ii);
        y = eps + (jj-1).*(1-2.*eps)./ii;
        plot(x,y)
    end
end

%% Load data
load('Matlab_data/Pfinal_delta.mat')

%% Create peak_locations from data


%% Plot cosine peaks - delta
delta = linspace(1,10);
eps = 1./(2.*delta);
n = n_mode(eps);
figure
hold on


plot(1./(2.*eps),1./(2.*eps),'k')
plot(1./(2.*eps),-1./(2.*eps),'k')

for i = 1:max(NoStates)
    idx = find(Location(:,1,i));
    plot(Value_delta(idx),2*Value_delta(idx).*(Location(idx,1,i)-0.5),'b.')
end

for i = 1:length(eps)
    m = 0:n(i);
    x = eps(i)+m.*(1-2.*eps(i))./n(i);
    %delta = 1/(2*eps(i));
    y = delta(i)*(2*x-1);
    plot(delta(i)*ones(1,length(m)),y,'r.')
end

%title('Location of peaks for fastest growing modes')
xlabel('\Delta')
ylabel('y')
xlim([1,10])
hold off
%% Save figures
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 12 10]);
saveas(gcf,'../Thesis_chapter/figs/bif_diag_delta.eps','epsc');

%% 
x = linspace(0,100,1000);
plot(x,abs(sin(x/3)./(x/3)))
hold on 
plot(x,abs(sin(2*x/3)./(2*x/3)))
plot(x,(sin(x/3)./(x/3) + sin(2*x/3)./(2*x/3))-sin(x)./x-1)
hold off
%%
n_ind = s_over_2eps(x_below)-s_over_2eps(x_above);

%%
s_over_2eps(x_below(eps==0.2))

%%Two classes
%% Plot of lambda as a function of x = eps*k
figure
colors = cell(1,3);
colors{1} = [0, 0.4470, 0.7410];
colors{2} = [0.8500, 0.3250, 0.0980];
colors{3} = [0.9290, 0.6940, 0.1250];
colors{4} = [0.4940, 0.1840, 0.5560];
%coeffs = [1,1;1,1]/4;
plots = [];
hold on
ii = 0;
for k_maj =[5,15,25,50]
    ii = ii+1;
degrees = [k_maj,100];
degree_dist = [0.9,0.1];
edge_probs = degrees'*degrees; %Not normalised by N<k>
q = degree_dist;
coeffs = edge_probs.*repmat(q,length(q),1);
coeffs = coeffs./(sum(q*coeffs));

x = linspace(0,5);
[y1,y2] = lambda_over_2eps(x,1/2,coeffs);
plots = [plots plot(x,y1, 'Color',colors{ii})];
x_max = x(find(y1==max(y1)));
y_max = max(y1);
plot(x,y2, 'Color',colors{ii})
line([x_max x_max], [0 y_max],'Color','black','LineStyle',':','Linewidth',0.8); 
end
%plot(x,y2)
ylabel('$$\lambda$$', 'Interpreter', 'LaTeX')
xlabel('$$\tilde{k}$$', 'Interpreter', 'LaTeX')
%y = s_over_2eps(x);
%plot(x,y,'b--')

ylim([-1 2])
plot(x,0*x,'k')
leg = legend([plots(1) plots(2) plots(3) plots(4)],'5','15','25','50');
title(leg,'k_{maj}')
%set(gca,'ytick',[])
%set(gca,'yticklabel',[])
hold off
% largest value of x is approx 2.8

%%
k_maj = 25;
eps = 0.25;
degrees = [k_maj,100];
degree_dist = [0.9,0.1];
edge_probs = degrees'*degrees; %Not normalised by N<k>
q = degree_dist;
coeffs = edge_probs.*repmat(q,length(q),1);
coeffs = coeffs./(sum(q*coeffs));

x = linspace(0,5,1000);

[y1,y2] = lambda_over_2eps(x,1/2,coeffs);

x_max = x(find(y1==max(y1)));
n = n_mode_lambda(eps,1/2,coeffs);

%% Save figures
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 10 8]);
saveas(gcf,'../../Thesis/4_Deffuant_MF_analysis/figs/lambda.eps','epsc');

%% Calculate max of lambda(x)
ii = 0;
x_max = zeros(1,4);
for k_maj =[5,25,50,100]
    ii = ii+1;
degrees = [k_maj,100];
degree_dist = [0.9,0.1];
edge_probs = degrees'*degrees; %Not normalised by N<k>
q = degree_dist;
coeffs = edge_probs.*repmat(q,length(q),1);
coeffs = coeffs./(sum(q*coeffs));
coeffs(1,1)*coeffs(2,2)-coeffs(1,2)*coeffs(2,1);
x = linspace(0.01,5,10000);
[lambda1,lambda2] = lambda_over_2eps(x,mu,coeffs); 
[val,idx] = max(lambda1);
x_max(ii) = x(idx);

end
    
%% Fastest growing modes
eps = linspace(0.05,0.5);
n = n_mode_lambda(eps,mu,coeffs);
plot(eps,n)
ylabel('n')
xlabel('\epsilon')

%% Calculate value of lambda(x) for eps = 0.05
eps = 0.2750;
n = n_mode_lambda(eps,mu,coeffs);
x = eps*2*pi*n/(1-2*eps);
[lambda1,lambda2] = lambda_over_2eps(x,mu,coeffs); 

%% Load data
load('Matlab_data/Pfinal_delta_k5_k100.mat')

%% Plot cosine peaks - delta
predict_classes = 1;
predict_orig = 0;
mu = 0.5;
degrees = [5,100];
degree_dist = [0.9,0.1];

%calculate coefficients
edge_probs = degrees'*degrees; %Not normalised by N<k>
q = degree_dist;
coeffs = edge_probs.*repmat(q,length(q),1);
coeffs = coeffs./(sum(q*coeffs));

delta = linspace(1,10,100);
eps = 1./(2.*delta);


figure
hold on

plot(1./(2.*eps),1./(2.*eps),'k')
plot(1./(2.*eps),-1./(2.*eps),'k')

for k = 1:length(Value_eps)
    J = length(P_final{k,1})/2;
    [M_maj,L_maj]=clusterloc(degree_dist(1)*P_final{k,1}(1:(J+1)),0.05); %cluster locations for the majority group - k = 5
    [M_min,L_min] = clusterloc(degree_dist(2)*P_final{k,1}((J+1):(2*J)),0.05);
    p1 = plot((1/(2.*Value_eps(k)))*ones(1,length(L_min)),(1/(2.*Value_eps(k))).*(2*L_min-1),'b.'); %Degree 100
    plot((1/(2.*Value_eps(k)))*ones(1,length(L_maj)),(1/(2.*Value_eps(k))).*(2*L_maj-1),'b.','MarkerSize',3) %Degree 5
end

if predict_classes == 1
    n = n_mode_lambda(eps,mu,coeffs);
    for i = 1:length(eps)
        m = 0:n(i);
        x = eps(i)+m.*(1-2.*eps(i))./n(i);
        %delta = 1/(2*eps(i));
        y = delta(i)*(2*x-1);
        p2 = plot(delta(i)*ones(1,length(m)),y,'r.');
    end
end

if predict_orig == 1
    coeffs = [1,1;1,1];
    delta = linspace(1.05,10.05,100);
    eps = 1./(2.*delta);
    n = n_mode_lambda(eps,mu,coeffs);
    for i = 1:length(eps)
        m = 0:n(i);
        x = eps(i)+m.*(1-2.*eps(i))./n(i);
        %delta = 1/(2*eps(i));
        y = delta(i)*(2*x-1);
        p2 = plot(delta(i)*ones(1,length(m)),y,'r.');
    end
end
%legend([p1,p2],{'Numerical solution','Approximation'})
%title('Location of final clusters')
xlabel('\Delta')
ylabel('y')
xlim([1,10])
hold off

%% Save figures
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 12 10]);
saveas(gcf,'../Thesis_chapter/figs/bif_diag_k5.eps','epsc');
%% Plot cluster mass
cluster_mass_mat = zeros(length(Value_eps),9);


for k = 1:length(Value_eps)
    ll = length(cluster_mass{k});
    if mod(ll,2)==0
        cluster_mass_mat(k,1) = 0;
        cluster_mass_mat(k,2:((ll/2)+1)) = cluster_mass{k}(1:(ll/2));
    else
        cluster_mass_mat(k,1) = cluster_mass{k}((ll+1)/2);
        cluster_mass_mat(k,2:((ll+1)/2)) = cluster_mass{k}(1:(ll-1)/2);
    end
end

%%
figure
hold on
plot(1./(2.*Value_eps),cluster_mass_mat(:,1));
for k = 2:9
    if mod(ll,2)==0
        plot(1./(2.*Value_eps),cluster_mass_mat(:,1)+ 2*sum(cluster_mass_mat(:,2:k),2));
    else 
        plot(1./(2.*Value_eps),cluster_mass_mat(:,1)+ 2*sum(cluster_mass_mat(:,2:k),2));
    end
end
hold off

%% Mass of minor clusters
figure
hold on
for k=[2,4,6,8]
    temp_clust = cluster_mass_mat(cluster_mass_mat(:,k)>0,k);
    temp_eps = Value_eps(cluster_mass_mat(:,k)>0);
    plot(1./(2.*temp_eps),temp_clust./temp_eps,'b')
end
ylim([0 0.6])
xlim([1,10])
xlabel('\Delta')
ylabel('m/\epsilon')
hold off

%% Save figures
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 8 6]);
saveas(gcf,'../Thesis_chapter/figs/mass_minor_k5.eps','epsc');

%% Mass of major clusters

colors1 = [0.0195, 0.1875, 0.3789];
colors2 = [0.3984, 0.7969, 0.9297];

figure
hold on
set(gca, 'YScale', 'log')
ind_clust = find(cluster_mass_mat(:,1)>0);
change_points = [0,find(diff(ind_clust)~=1)',length(ind_clust)];
for ii = 1:(length(change_points)-1)
    temp_points = ind_clust((change_points(ii)+1):change_points(ii+1));
    temp_clust = cluster_mass_mat(temp_points,1);
    temp_eps = Value_eps(temp_points);
    plot(1./(2.*temp_eps),temp_clust./temp_eps,'--','Color',colors1)
end


for k=[3,5,7,9]
    temp_clust = cluster_mass_mat(cluster_mass_mat(:,k)>0,k);
    temp_eps = Value_eps(cluster_mass_mat(:,k)>0);
    plot(1./(2.*temp_eps),temp_clust./temp_eps,'Color',colors1)
end
%ylim([0 3.5])
xlim([1,10])

xlabel('\Delta')
ylabel('m/\epsilon')

% Include minor clusters
for k=[2,4,6,8]
    temp_clust = cluster_mass_mat(cluster_mass_mat(:,k)>0,k);
    temp_eps = Value_eps(cluster_mass_mat(:,k)>0);
    plot(1./(2.*temp_eps),temp_clust./temp_eps,'Color',colors2)
end

hold off

%% Save figures
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 8.6 8]);
saveas(gcf,'../../Thesis/4_Deffuant_MF_analysis/figs/mass_k5.eps','epsc');

%% Save figures
set(gcf, 'PaperUnits', 'centimeters');
set(gcf, 'PaperPosition', [0 0 8 6]);
saveas(gcf,'../Thesis_chapter/figs/mass_major_k5.eps','epsc');

%%
x = 2*pi/3;
lambda_over_2eps(x,mu,coeffs)
%%
   x = linspace(0.01,5,10000);
    [lambda1,lambda2] = lambda_over_2eps(x,mu,coeffs); 
    [val,idx] = max(lambda1);
    x_max = x(idx);
    %%
    n_max = (1-2.*eps).*x_max./(eps.*2.*pi); % noninteger value of n correponding to x = 2.8 (max of s/2eps)
    x_below = 2.*floor(n_max).*pi.*eps./(1-2.*eps); %value of x corresponding to the closest integer value of n below n_max
    x_above = 2.*ceil(n_max).*pi.*eps./(1-2.*eps); %value of x corresponding to the closest integer value of n above n_max

    n_ind = s_over_2eps(x_below)> s_over_2eps(x_above); %cases where n below n_max has a bigger value of s than n above n_max
    n(n_ind) = floor(n_max(n_ind));
    n(~n_ind) = ceil(n_max(~n_ind));
    n = double(n);
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

function[l1,l2] = lambda_over_2eps(x,mu,coeffs)
    l1 = zeros(size(x));
    l2 = zeros(size(x));
    x_old = x;
    x= x(x~=0);
    f = sin(mu.*x)./(mu.*x) - 1;
    g = sin((1-mu).*x)./((1-mu).*x) - sin(x)./(x);
    l1(x_old~=0) = sum(sum(coeffs)).*f + (coeffs(1,1)+coeffs(2,2)).*g...
        + sqrt(((coeffs(1,1)+coeffs(1,2)-coeffs(2,1)-coeffs(2,2)).*f + (coeffs(1,1)-coeffs(2,2)).*g).^2 +4*coeffs(1,2)*coeffs(2,1)*g.^2);
    l2(x_old~=0) = sum(sum(coeffs)).*f + (coeffs(1,1)+coeffs(2,2)).*g...
        - sqrt(((coeffs(1,1)+coeffs(1,2)-coeffs(2,1)-coeffs(2,2)).*f + (coeffs(1,1)-coeffs(2,2)).*g).^2 +4*coeffs(1,2)*coeffs(2,1)*g.^2);
end

function[n] = n_mode_lambda(eps,mu,coeffs)
    %function to calculate the fastest growing mode for a given eps
    %calculate x_max

    x = linspace(0.01,5,10000);
    [lambda1,lambda2] = lambda_over_2eps(x,mu,coeffs); 
    [val,idx] = max(lambda1);
    x_max = x(idx);
    
    n_max = (1-2.*eps).*x_max./(eps.*2.*pi); % noninteger value of n correponding to x = 2.8 (max of s/2eps)
    x_below = 2.*floor(n_max).*pi.*eps./(1-2.*eps); %value of x corresponding to the closest integer value of n below n_max
    x_above = 2.*ceil(n_max).*pi.*eps./(1-2.*eps); %value of x corresponding to the closest integer value of n above n_max

    n_ind = s_over_2eps(x_below)> s_over_2eps(x_above); %cases where n below n_max has a bigger value of s than n above n_max
    n(n_ind) = floor(n_max(n_ind));
    n(~n_ind) = ceil(n_max(~n_ind));
    n = double(n);
end