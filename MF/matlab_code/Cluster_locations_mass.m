%% Mass and location of clusters for each final distribution and number of final states
filename = 'Pfinal_delta_k5_k100.mat';
load(filename)

P_final_delta = P_final;%(21:91,1);
Value_delta = 1./(2.*Value_eps);
[n_eps,n_mu] = size(P_final_delta);
N_clust_max = floor(2*max(Value_delta))+1;
%n_eps = 250;
Mass = zeros(n_eps,n_mu,N_clust_max);
Location = zeros(n_eps,n_mu,N_clust_max);
threshold = 0.05;

for itermu =1:n_mu
    for itereps =1:n_eps
        [M,L] =clusterloc(P_final_delta{itereps,itermu},threshold);  
        %M = cluster_mass{itereps+20,itermu};
        %L = cluster_loc{itereps+20,itermu};
        Mass(itereps,itermu,1:length(M))= M;
        Location(itereps,itermu,1:length(M))=L; 
    end
end

NoStates = zeros(n_eps,n_mu);
for itermu = 1:n_mu
    for itereps = 1:n_eps
        NoStates(itereps,itermu) = nnz(Mass(itereps,itermu,:));
    end
end

%%
%save('.\Matlab_data\Pfinal_delta.mat','P_final_delta','Value_delta','Mass','Location','NoStates')
% save(filename,'P_final','Value_eps','degrees','degree_dist','cluster_loc','cluster_mass','t_final','Mass','Location','NoStates','Value_delta');

%% DBMF
filename = 'Pfinal_delta_k5_k100.mat';
load(filename)

P_final_delta = P_final;%(21:91,1);
Value_delta = 1./(2.*Value_eps);
[n_eps,n_mu] = size(P_final_delta);
N_clust_max = floor(2*max(Value_delta))+1;
%n_eps = 250;
Mass = zeros(n_eps,2,N_clust_max);
Location = zeros(n_eps,2,N_clust_max);
threshold = 0.05;


    for itereps =1:n_eps
        k = itereps;
        J = length(P_final{k,1})/2;
        [M_maj,L_maj]=clusterloc(degree_dist(1)*P_final{k,1}(1:(J+1)),0.05); %cluster locations for the majority group - k = 5
        [M_min,L_min] = clusterloc(degree_dist(2)*P_final{k,1}((J+1):(2*J)),0.05);
        %[M,L] =clusterloc(P_final_delta{itereps,itermu},threshold);  
        %M = cluster_mass{itereps+20,itermu};
        %L = cluster_loc{itereps+20,itermu};
        Mass(itereps,1,1:length(M_maj))= M_maj;
        Mass(itereps,2,1:length(M_min))= M_min;
        Location(itereps,1,1:length(L_maj))=L_maj;
        Location(itereps,2,1:length(L_min))=L_min;
    end

% 
% NoStates = zeros(n_eps,n_mu);
% for itermu = 1:n_mu
%     for itereps = 1:n_eps
%         NoStates(itereps,itermu) = nnz(Mass(itereps,itermu,:));
%     end
% end
%%
% save(filename,'P_final','Value_eps','degrees','degree_dist','cluster_loc','cluster_mass','t_final','Mass','Location','Value_delta');


% %% Load data
% load('Pfinal.mat')

%% Plot of cluster locations vs epsilon
[n_eps,n_mu] = size(P_final);
for i = 1:n_eps
    plot(Value_eps(i,n_mu)*ones(NoStates(i),1),Location{i,n_mu},'b.')
    hold on
end
plot(Value_eps,1-1.25*Value_eps)
xlabel('\epsilon')
ylabel('x')
title('Cluster locations for varying \epsilon (\mu = 1/2)')
hold off

%% Plot of cluster locations vs delta
% This is what we need, plots biffurcation diagramm
% hold on
max_mass_value = max(Mass(:));
for i = 1:max(NoStates)
    idx = find(Location(:,1,i));
%     plot(Value_delta(idx),2*Value_delta(idx).*(Location(idx,1,i)-0.5),'b.')
    plot(Value_delta(idx),(Mass(idx,1,i))/max_mass_value,'b.')
    hold on
%       plot(1./Value_delta(idx)/2,(Mass(idx,1,i)).*1./Value_delta(idx)/2,'b.')
end
xline([1.7387,  2.7387, 3.8468,  4.9640,  6.0901,  7.2072,  8.3333,  9.4595], '--r', 'LineWidth',1.5)
xlabel('\Delta')
ylabel('x')
title('Cluster masses as a function of \Delta')
set(gca,'Fontsize', 18)
hold off

%% 
%     i =5;
%     idx = find(Location(:,1,i));
%     plot(Value_delta(idx),(Mass(idx,1,i)),'b.')
% %     plot(Value_delta(idx),(Mass(idx,1,i)),'b.')
%% Plot - final distribution
n = length(PfinalCTDS{1,1})-1;
figure
plot(0:1/n:1,PfinalCTDS{1,1})
xlabel('x')
ylabel('P(x)')
title('Final distribution, eps=0.05,mu=0.05')

%% Plot - different location of clusters for different values of mu
n = length(PfinalCTDS{1,1})-1;
xcoords = 225/n:1/n:275/n;
ycoords = 225:275;
figure
plot(xcoords,PfinalCTDS{1,10}(ycoords))
hold on
plot(xcoords,PfinalCTDS{1,5}(ycoords))
hold on
plot(xcoords,PfinalCTDS{1,1}(ycoords))
hold on
plot(xcoords,P(end,ycoords))
%hold on
%plot(xcoords,PfinalCTDS{1,9}(ycoords))
legend('\mu=0.5','\mu=0.25','\mu=0.05','\mu=0.025')%,'\mu=0.7','\mu=0.9')
title('Location of a cluster for different values of \mu (\epsilon = 0.05)')
xlabel('x');
ylabel('P(x)');
hold off

%% Plot - comparing final distributions for different values of mu
figure
subplot(3,1,1)
semilogy(PfinalCTDS{1,9})
subplot(3,1,2)
semilogy(PfinalCTDS{1,8})
subplot(3,1,3)
semilogy(PfinalCTDS{1,7})

%Convergence Time
figure
xvalues = 0.1:0.1:0.9;
yvalues = 1:-0.1:0.1;
heatmap(xvalues,yvalues,log(TCTDS(10:-1:1,:)),'CellLabelColor','none')
title('Convergence Time');
xlabel('Multiplier (mu)');
ylabel('Confidence Bound (epsilon)');
%% Number of clusters
figure
xvalues = 0.1:0.1:0.9;
yvalues = 1:-0.1:0.1;
heatmap(xvalues,yvalues,NoStatesCTDS(20:-2:2,2:2:18),'CellLabelColor','none')
colormap(jet(9))
ylabel('Confidence bound (epsilon)')
xlabel('Multiplier (mu)')
title('Number of clusters in steady state')

%% Number of clusters
%figure
x = 0.05:0.05:0.95;
y = 0.05:0.05:1;
imagesc(x,y,NoStatesCTDS); 
%colormap jet; 
colormap(jet(15)); 
axis xy;
colorbar
xlabel('\mu')
ylabel('\epsilon')
title('Number of clusters in steady state')


 
 
%% Number of clusters
x = 0.01:0.01:0.5;
y = 0.05:0.01:5;
imagesc(x,y,NoStatesCTDSnew); 
%colormap jet; 
axis xy;

%% Number of clusters
figure
t_plot = TCTDSnew(5:50,:);
m_plot = NoStatesCTDSnew;
m_plot(t_plot == 4e3)=20;
imagesc(x,y,m_plot); 
colormap(jet(20))
axis xy;
ylabel('Confidence bound (epsilon)')
xlabel('Multiplier (mu)')
title('Number of clusters in steady state')
 
 
%% Plots
P_density_eps03_mu05_new = P*(n+1);
t_eps03_mu05_new = t;
%save('MF_eps03_mu05_new.mat','P_density_eps03_mu05_new','t_eps03_mu05_new')


%bin_size = 1/n;
%bin_centers = (bin_size/2):bin_size:(1-(bin_size/2));

%%
n = 667;
x = (0:(n-1))/(n-1);
semilogy(x,P_density_eps03_mu015(53,:))
line(x,P_density(end,:),'color','red')

%%
%figure
eps = 1;
mu = 10;
y = MassCTDS{eps,mu};
x = LocationCTDS{eps,mu};
plot(x,y,'s');
set(gca,'TickLength',[0 0.1])

xlabel('x')
ylabel('Mass')
title('Mass of clusters for varying \mu (\epsilon = 0.05)');
ylim([-0.001 0.16])
%ylim([0.1 0.13])
hold on

mu = 5;
y = MassCTDS{eps,mu};
x = LocationCTDS{eps,mu};
plot(x,y,'v');

mu = 1;
y = MassCTDS{eps,mu};
x = LocationCTDS{eps,mu};

plot(x,y,'.','MarkerSize',10);

x = L;
y=M;
plot(x,y,'*');

legend('\mu=0.5','\mu=0.25','\mu=0.05','\mu=0.025');

hold off


%%
%figure
eps = 1;
mu = 10;
y = MassCTDS{eps,mu};
x = LocationCTDS{eps,mu};
semilogy(x,y,'.','MarkerSize',10);

xlabel('Location')
ylabel('Mass')
title('Mass of clusters for varying \mu (\epsilon = 0.05)');
ylim([0 0.16])
hold on

mu = 5;
y = MassCTDS{eps,mu};
x = LocationCTDS{eps,mu};
semilogy(x,y,'.','MarkerSize',10);

mu = 1;
y = MassCTDS{eps,mu};
x = LocationCTDS{eps,mu};
semilogy(x,y,'.','MarkerSize',10);

x = L;
y=M;
semilogy(x,y,'.','MarkerSize',10);

legend('\mu=0.5','\mu=0.25','\mu=0.05','\mu=0.025');

hold off
%%
n = length(PfinalCTDS{1,1});
x = (0:(n-1))/(n-1);
plot(x,PfinalCTDS{1,1});
xlabel('x')
ylabel('P(x)')
title('P(x) once final clusters have formed (\epsilon = 0.05, \mu = 0.05)')
set(gca,'TickLength',[0 0.1])

%%

n = length(P(1,:));
x = (0:(n-1))/(n-1);
plot(x,P(250,:));

%% Bigger cell
n_eps_old = 25;
n_mu_old = 45;
n_eps = 50;
n_mu = 50;
Ptemp = cell(n_eps,n_mu);
Ttemp = zeros(n_eps,n_mu);
Etemp = zeros(n_eps,n_mu);
Mtemp = zeros(n_eps,n_mu);
index_shift_eps = 24;
index_shift_mu = 4;
for i=1:n_eps_old
    for j =1:n_mu_old
        Ptemp{i+index_shift_eps,j+index_shift_mu} = PfinalCTDSnew{i,j};
        Ttemp(i+index_shift_eps,j+index_shift_mu) = TCTDSnew(i,j);
        Etemp(i+index_shift_eps,j+index_shift_mu) = ValueEpsCTDSnew(i,j);
        Mtemp(i+index_shift_eps,j+index_shift_mu) = ValueMuCTDSnew(i,j);
    end
end

PfinalCTDSnew = Ptemp;
TCTDSnew = Ttemp;
ValueEpsCTDSnew = Etemp;
ValueMuCTDSnew = Mtemp;

%% eps=0.35,mu=0.2
%Convergence Time
figure
xvalues = 0.01:0.01:0.50;
yvalues = 0.50:-0.01:0.05;
%heatmap(xvalues,yvalues,log(TCTDSnew(15:-1:1,:)),'CellLabelColor','none')
T = TCTDSnew;
T(T==4e1) = 600;
t_plot = T(50:-1:5,:);
heatmap(xvalues,yvalues,log(t_plot),'CellLabelColor','none')
title('Convergence Time');
xlabel('Multiplier (mu)');
ylabel('Confidence Bound (epsilon)');

%% Number of clusters
figure

m_plot = NoStatesCTDSnew(46:-1:1,:);
m_plot(t_plot == 4e3)=20;
heatmap(xvalues,yvalues,m_plot,'CellLabelColor','none')
colormap(jet(20))
ylabel('Confidence bound (epsilon)')
xlabel('Multiplier (mu)')
title('Number of clusters in steady state')

%%
mu=0.5;
eps=0.3;
threshold = 1e-5;
eps_state=200;
tmax = 100;
tvect = 1:30;
specific_times = 1;
degrees = [100,50];
degree_dist = [0.1,0.9];
n=floor(eps_state/eps)-1;
x = (0:1:n)/n;
%y1 = normpdf(x,0.25,0.15);
%y2 = normpdf(x,0.75,0.15);
P0 = ones((n+1)*length(degrees),1)/(n+1); %P0,P are probability distributions - P0*n,P*n are probability densities.
%P0 = [y1/sum(y1),y2/sum(y2)];

tic
[t,P,te,ye,ie] = DeffuantfcnCTDSdegree(mu,eps,threshold,eps_state,tmax,tvect,specific_times,degrees,degree_dist,P0);
toc

%%
P = P*(n+1);
save('../../Data/MF_eps03_mu05_k100_01_k25_09.mat','P','t')
%%
%P_init = degree_dist(1)*P0(1:(n+1))+degree_dist(2)*P0((n+2):2*(n+1));
P_init = P0;
%Pnorm = degree_dist(1)*P(:,1:(n+1))+degree_dist(2)*P(:,(n+2):2*(n+1));
Pnorm = P;
tnorm = t;
save('MF_eps025_mu05_norm_deg.mat','Pnorm','tnorm')
%%
mu=0.5;
eps=0.3;
threshold = 1e-5;
eps_state=200;
tmax = 100;
tvect = 1:10;
specific_times = 1;
degrees = 1;
degree_dist = 1;
n=floor(eps_state/eps)-1;
x = (0:1:n)/n;
%y1 = normpdf(x,0.25,0.15);
%y2 = normpdf(x,0.75,0.15);
%y = 0.1*y1/sum(y1)+0.9*y2/sum(y2);
%P0 = ones((n+1)*length(degrees),1)/(n+1); %P0,P are probability distributions - P0*n,P*n are probability densities.
%P0 = y;
y1 = m*x+1-(m/2);
P0 = y1/sum(y1);

tic
[t,P,te,ye,ie] = DeffuantfcnCTDSdegree(mu,eps,threshold,eps_state,tmax,tvect,specific_times,degrees,degree_dist,P0);
toc

%% Fully mixed
mu=0.5;
eps=0.3;
threshold = 1e-5;
eps_state=200;
tmax = 100;
tvect = 1:10;
specific_times = 1;
n_degrees = 1;

[t,P,te,ye,ie] = DeffuantfcnCTDS(mu,eps,threshold,eps_state,tmax,tvect,specific_times,n_degrees);


%%
Pnorm = P;
tnorm = t;
save('MF_eps025_mu05_norm.mat','Pnorm','tnorm')
%%
t_plot = 14;
figure
plot(Pnew(t_plot,:))
title(['t = ',num2str(t_plot)])
hold on
plot(P_deg(t_plot,1:800)*0.1+0.9*P_deg(t_plot,801:1600))

%%
x = (0:1:n)/n;
y = normpdf(x,0.25,0.05);
plot(x,y)

%%
figure
semilogy(0.1*P1(end,1:800)+0.9*P1(end,801:1600))
hold on
semilogy(P(end,:))

[M,L] = clusterloc(P(end,:),threshold)
[M1,L1] = clusterloc(0.1*P1(end,1:800)+0.9*P1(end,801:1600),threshold)