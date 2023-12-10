%% One degree class
degrees = 1;
degree_dist = 1;
eps = 0.3;

J = 1000;
tspan = eps*(0:0.1:50);
P0 = ones(J,1);
tic()
[t_bnd_deg,P_bnd_deg] = ode113(@(t,P) odefcnbndregs(t,P, degrees, degree_dist ),tspan,P0);
toc()

%%
figure
t_plot = 9;
x = ((0:(J-1)) +(1:J))/(2*J);
plot(x,P_bnd_deg(t_plot,:))
title(['t = ',num2str(t_bnd_deg(t_plot))])

%% Load data 
load('../Data/MF_eps03_mu05_early.mat','P','Ptot','t','degrees')
P_num_deg = P;
Ptot_num_deg = Ptot;
t_num_deg = t;

%% Comparison with numerics - boundary regions
%Numerics

color1 = [0.9290, 0.6940, 0.1250];
t_plot = 30;
eps = 0.3;
n = length(P_num_deg(1,:))/length(degrees);
x_max  = ceil(eps*n);
x = ((0:(x_max - 1))+ (1:(x_max)))/(2*n);
plot(x,P_num_deg(t_plot,(1:(x_max))))
title(['t = ',num2str(t_num_deg(t_plot))])

hold on
n = length(P_bnd_deg(1,:))/length(degrees);
x_max  = n;
x = eps*((0:(x_max - 1))+ (1:(x_max)))/(2*n);
plot(x,P_bnd_deg(t_plot,(1:x_max)))
legend("Original","Boundary")

% 
% t1 = t_plot/10;
% x1 = linspace(0,eps/2);
% plot(x1,1./(1-(3*x1-eps)*t1),'--','Linewidth',2,'Color',color1)
% 
% x1 = linspace(eps/2,eps);
% plot(x1,1./(1-(eps-x1)*t1),'--','Linewidth',2,'Color',color1)
% 
% x1 = linspace(eps,1-eps);
% plot(x1,ones(100,1),'--','Linewidth',2,'Color',color1)
% 
% x1 = linspace(1-eps,1-eps/2);
% plot(x1,1./(1-(eps-(1-x1))*t1),'--','Linewidth',2,'Color',color1)
% 
% x1 = linspace(1-eps/2,1);
% plot(x1,1./(1-(3*(1-x1)-eps)*t1),'--','Linewidth',2,'Color',color1)

hold off

%% Comparison with numerics - whole domain without analytic
%Numerics
plot_times = 5*[10,20,30]+1;
k=0;
for t_plot = plot_times
    k = k+1;
    subplot(length(plot_times),1,k)
eps = 0.3;
n = length(P_num_deg(1,:))/length(degrees);
x_max  = n;
x = ((0:(x_max - 1))+ (1:(x_max)))/(2*n);
p1 =plot(x,P_num_deg(t_plot,(1:(x_max))), 'k');
title(['t = ',num2str(t_num_deg(t_plot))])


hold on
n = length(P_bnd_deg(1,:))/length(degrees);
x_max  = n;
x = eps*((0:(x_max - 1))+ (1:(x_max)))/(2*n);
p2 =plot(x,P_bnd_deg(t_plot,(1:x_max)), 'r--');
plot(1-x(end:-1:1),P_bnd_deg(t_plot,(x_max:-1:1)),'r--')
plot(linspace(eps,1-eps),ones(1,100),'r--')

legend([p1,p2],'Num orig','Num bnd')
hold off
end


%% Comparison with numerics - whole domain with analytic
%Numerics
plot_times = 5*[10,20,30]+1;
k=0;
for t_plot = plot_times
    k = k+1;
    subplot(length(plot_times),1,k)
eps = 0.3;
n = length(P_num_deg(1,:))/length(degrees);
x_max  = n;
x = ((0:(x_max - 1))+ (1:(x_max)))/(2*n);
p1 =plot(x,P_num_deg(t_plot,(1:(x_max))), 'k');
title(['t = ',num2str(t_num_deg(t_plot))])

hold on
n = length(P_bnd_deg(1,:))/length(degrees);
x_max  = n;
x = eps*((0:(x_max - 1))+ (1:(x_max)))/(2*n);
p2 =plot(x,P_bnd_deg(t_plot,(1:x_max)), 'r--');
plot(1-x(end:-1:1),P_bnd_deg(t_plot,(x_max:-1:1)),'r--')
plot(linspace(eps,1-eps),ones(1,100),'r--')

t1 = (t_plot-1)/10;
x1 = linspace(0,eps/2);
p3 =plot(x1,1./(1-(3*x1-eps)*t1),'b--');

x1 = linspace(eps/2,eps);
plot(x1,1./(1-(eps-x1)*t1),'b--')

x1 = linspace(eps,1-eps);
plot(x1,ones(100,1),'b--')

x1 = linspace(1-eps,1-eps/2);
plot(x1,1./(1-(eps-(1-x1))*t1),'b--')

x1 = linspace(1-eps/2,1);
plot(x1,1./(1-(3*(1-x1)-eps)*t1),'b--')

legend([p1,p2,p3],'Num orig','Num bnd','Analytic')
hold off
end

%% Multiple degree classes
degrees = [5,100];
degree_dist = [0.9,0.1];
eps = 0.3;

J = 1000;
tspan = eps*(0:0.1:100);
P0 = ones(length(degrees)*J,1);
tic()
[t_bnd_deg,P_bnd_deg] = ode113(@(t,P) odefcnbndregs(t,P, degrees, degree_dist ),tspan,P0);
toc()




%%
figure
plot(sum(P_bnd_deg(:,1:J),2)/J)
%%
plot(sum(P_bnd_deg(:,(J+1):(2*J)),2)/J)
%% Load data 
load('../Data/MF_eps03_mu05_k100_01_k5_09_early.mat','P','Ptot','t','degrees')
P_num_deg = P;
Ptot_num_deg = Ptot;
t_num_deg = t;

%% Comparison with numerics - boundary regions
%Numerics
t_plot = 2;
eps = 0.3;
n = length(P_num_deg(1,:))/length(degrees);
x_max  = ceil(eps*n);
x = ((0:(x_max - 1))+ (1:(x_max)))/(2*n);
plot(x,P_num_deg(t_plot,n+(1:(x_max))), 'Color','b')
title(['t = ',num2str(t_num_deg(t_plot))])

hold on
plot(x,P_num_deg(t_plot,(1:(x_max))), 'Color','r')
title(['t = ',num2str(t_num_deg(t_plot))])

%Boundary
n = length(P_bnd_deg(1,:))/length(degrees);
x_max  = n;
x = eps*((0:(x_max - 1))+ (1:(x_max)))/(2*n);
plot(x,P_bnd_deg(t_plot,n+(1:x_max)),'--b')


plot(x,P_bnd_deg(t_plot,(1:x_max)),'--r')
hold off

%% Comparison with numerics - whole domain
%Numerics
plot_times = 2*[10,20,30]+1;
k=0;
for t_plot = plot_times
    k = k+1;
    subplot(length(plot_times),1,k)
eps = 0.3;
n = length(P_num_deg(1,:))/length(degrees);
x_max  = n;
x = ((0:(x_max - 1))+ (1:(x_max)))/(2*n);

p1 = semilogy(x,P_num_deg(t_plot,n+(1:(x_max))), 'Color','b');
title(['t = ',num2str(t_num_deg(t_plot))])
%ylim([0,0.1])
hold on
plot(x,P_num_deg(t_plot,(1:(x_max))), 'Color','r')
title(['t = ',num2str(t_num_deg(t_plot))])

%Boundary
n = length(P_bnd_deg(1,:))/length(degrees);
x_max  = n;
x = eps*((0:(x_max - 1))+ (1:(x_max)))/(2*n);
p2 = plot(x,P_bnd_deg(t_plot,n+(1:x_max)),'--b');
plot(1-x(end:-1:1),P_bnd_deg(t_plot,n+(x_max:-1:1)),'--b')


plot(x,P_bnd_deg(t_plot,(1:x_max)),'--r')
plot(1-x(end:-1:1),P_bnd_deg(t_plot,(x_max:-1:1)),'--r')
hold off

legend([p1,p2],'Original','Boundary')
end

%% Comparison with numerics - whole domain - deg100
%Numerics
plot_times = 5*[10,20,30]+1;
k=0;
for t_plot = plot_times
    k = k+1;
    subplot(length(plot_times),1,k)
eps = 0.3;
n = length(P_num_deg(1,:))/length(degrees);
x_max  = n;
x = ((0:(x_max - 1))+ (1:(x_max)))/(2*n);

p1 = plot(x,P_num_deg(t_plot,n+(1:(x_max))), 'Color','b');
title(['t = ',num2str(t_num_deg(t_plot))])
%ylim([0,0.1])
hold on
%plot(x,P_num_deg(t_plot,(1:(x_max))), 'Color','r')
%title(['t = ',num2str(t_num_deg(t_plot))])

%Boundary
n = length(P_bnd_deg(1,:))/length(degrees);
x_max  = n;
x = eps*((0:(x_max - 1))+ (1:(x_max)))/(2*n);
p2 = plot(x,P_bnd_deg(t_plot,n+(1:x_max)),'--b');
plot(1-x(end:-1:1),P_bnd_deg(t_plot,n+(x_max:-1:1)),'--b')
ylim([0,0.1])

%plot(x,P_bnd_deg(t_plot,(1:x_max)),'--r')
%plot(1-x(end:-1:1),P_bnd_deg(t_plot,(x_max:-1:1)),'--r')
hold off

legend([p1,p2],'Original','Boundary')
end

%% Comparison with numerics - whole domain - deg5
%Numerics
plot_times = 10*[10,20,30]+1;
k=0;
for t_plot = plot_times
    k = k+1;
    subplot(length(plot_times),1,k)
eps = 0.3;
n = length(P_num_deg(1,:))/length(degrees);
x_max  = n;
x = ((0:(x_max - 1))+ (1:(x_max)))/(2*n);



p1 = plot(x,P_num_deg(t_plot,(1:(x_max))), 'Color','r');
title(['t = ',num2str(t_num_deg(t_plot))])
hold on
%Boundary
n = length(P_bnd_deg(1,:))/length(degrees);
x_max  = n;
x = eps*((0:(x_max - 1))+ (1:(x_max)))/(2*n);
%p2 = plot(x,P_bnd_deg(t_plot,n+(1:x_max)),'--b');
%plot(1-x(end:-1:1),P_bnd_deg(t_plot,n+(x_max:-1:1)),'--b')

ylim([0,0.1])
p2 = plot(x,P_bnd_deg(t_plot,(1:x_max)),'--r');
plot(1-x(end:-1:1),P_bnd_deg(t_plot,(x_max:-1:1)),'--r')
hold off

legend([p1,p2],'Original','Boundary')
end