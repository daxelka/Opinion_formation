%% Set parameters
degrees = [25,100];
degree_dist = [0.9,0.1]; %proportion of nodes within each class

mu = 0.5;
eps = 0.3;
threshold = 0.002/eps; %Used when deciding whether the distribution has converged (see DeffuantEvents.m)
eps_state = 200; %This is the number of states in a distance epsilon, eps_state/eps states altogther
n=800;
tspan = 0:1:10;

%% class based MF
%Initial distribution
P0 = ones(1,(floor(eps_state/eps)).*length(degree_dist)); %P0,P are probability densities.

%Calculate matrix of coefficients for CBMF
q = degree_dist;
pi_cbmf = degrees'*degrees; %Not normalised by N<k>
coeffs = pi_cbmf.*repmat(q,length(q),1);
coeffs = coeffs./(sum(q*coeffs));

tic()
[t,P,te,ye,ie] =DeffuantfcnGen(mu,eps,threshold,eps_state,tspan,coeffs,q,P0);
toc()

% Get total distribution
n = length(P(1,:))/length(q);
Ptot = zeros(length(P(:,1)),n);
for i = 1:length(q)
    Ptot = Ptot + P(:,((i-1)*n+1):(i*n))*q(i);
end

% Plotting
plot(Ptot(end,:))
%% Save data
% save(strcat('../Data/mf_dbmf_k',strrep(num2str(degrees(1)),'.',''),'.mat'),'P','Ptot','t','degrees','degree_dist','coeffs')