%% Set parameters
font = 14;
T = 30; start = 8; shift = 4;
col = parula(T-start+1 + shift);

L = 20;

% for full model
eps = 0.1;
eps_state = 50; %This is the number of states in a distance epsilon, eps_state/eps states altogther

tspan = 0:1:T;

    %Initial distribution
    P0 = ones(1,(eps_state*L)); %P0,P are probability densities.

    tic()
    [t,P] = ode45(@(t,P) odefcnDeffuantMu05_asymptotics(t,P,L,1),tspan,P0);
    toc()

    x = linspace(0,L,length(P(end,:)));

figure(1)
for i = start:length(t)
plot(x,P(i,:)/max(P(i,:)),'color',col(length(t)-i+1,:)); hold on;
end
hold off; xlim([0,10]); shg;

set(gca,'fontsize',font);
xlabel('$\tilde{x}=x/\epsilon$','interpreter','latex','fontsize',font);
ylabel('$\tilde{P}/\max(\tilde{P})$','interpreter','latex','fontsize',font);