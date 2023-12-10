function [t,P,te,ye,ie] = DeffuantfcnGen(mu,eps,threshold,eps_state,tspan,coeffs,q,P0)

degree_dist = q;
options = odeset('Events',@(t,P) DeffuantEvents(t,P,threshold,eps_state,degree_dist),'NonNegative',1:length(P0));

if mu==0.5
    [t,P,te,ye,ie] = ode113(@(t,P) odefcnDeffuantMu05(t,P,eps,coeffs),tspan,P0,options);
else
    %If mu!=0.5 then interaction could lead to individuals falling between
    %states. If this happens they move to the two nearest states with
    %probabilites proportional to the distance from each.
    I = cell(n+1,n+1); %I(i,k) will list the states that interact with i to reach state k
    TransProbs = cell(n+1,n+1);
    for i =0:n
        for k = 0:n
            I{i+1,k+1} = max([floor(i-eps_state)+1,0,floor((k-i-1)/mu+i)+1]):min([ceil(i+eps_state)-1,n,ceil((k-i+1)/mu+i)-1]);
            TransProbs{i+1,k+1} = 1-abs(k-i-mu*(I{i+1,k+1}-i));
        end
    end

    [t,P,te,ye,ie] = ode113(@(t,P) odefcnDeffuantGen(t,P,I,TransProbs,eps_state,mu,coeffs),tspan,P0,options);
end