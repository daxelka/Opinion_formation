function dydt = odefcnDeffuantMu05(t,P,eps ,coeffs)

n_classes = length(coeffs(1,:));
n = length(P)/n_classes;
P_temp = reshape(P,n,n_classes)';
B_temp = zeros(n_classes,n,n);
eps_state = ceil(eps*n);

%In the one class case, the transition from state j to state i is B_ji =
%0.5*P_jP_{2i-j-1} + P_jP_{2i-j} + 0.5*P_jP_{2i-j+1}. Construct vectors so
%we can calculate row j in one step (rather than looping through i)

%N's are vectors of 0's and 1's and index the states that can interact
%(need x_i-x_j<epsilon for interaction)
Nl = repmat([zeros(1,n-floor(eps_state/2)),ones(1,2*floor(eps_state/2)),zeros(1,n-1-floor(eps_state/2))],n_classes,1);
Nu = repmat([zeros(1,n-floor(eps_state/2)-1),ones(1,2*floor(eps_state/2)),zeros(1,n-floor(eps_state/2))],n_classes,1);
N0 = repmat([zeros(1,n-ceil(eps_state/2)),ones(1,2*ceil(eps_state/2)-1),zeros(1,n-ceil(eps_state/2))],n_classes,1);

v = [zeros(n_classes,n),P_temp,zeros(n_classes,n)];
vu = v(:,3:(length(v)))/2; 
v0 = v(:,2:(length(v)-1));
vl = v(:,1:(length(v)-2))/2;
for j = 1:(n)
    %Working with transpose: B_temp(k,:,j) is the jth row of class k
    %(needed for working with 3 dimensional arrays in matlab)
    B_temp(:,:,j) = (1/n).*(vl(:,(n+1-j):2:(3*n-1-j)).*Nl(:,(n-j+1):(2*n-j))...
        +vu(:,(n+1-j):2:(3*n-1-j)).*Nu(:,(n-j+1):(2*n-j))+v0(:,(n+1-j):2:(3*n-1-j)).*N0(:,(n-j+1):(2*n-j))); %Construct transition matrix row by row
    B_temp(:,j,j) = zeros(n_classes,1);
end

%Set B_temp(j,:,k) to be the jth row of class k
B_temp = permute(B_temp,[3,2,1]);

for k = 1:n_classes
    B_temp(:,:,k) = B_temp(:,:,k) + diag(-sum(B_temp(:,:,k),2),0);
end

for k = 1:n_classes
    B = zeros(n,n);
    for l =1:n_classes
        B = B + coeffs(k,l).*B_temp(:,:,l);
    end
    dydt1(((k-1)*n+1):((k)*n)) = P_temp(k,:)*B;
end
dydt = dydt1';