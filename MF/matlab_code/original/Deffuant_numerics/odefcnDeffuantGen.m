function dydt = odefcnDeffuantGen(t,P,I,TransProbs,eps_state,mu,coeffs)
%P assumed to be a density - need a factor of 1/n on line 10.
n_classes = length(coeffs(1,:));
n = length(P)/n_classes;
P_temp = reshape(P,n,n_classes)';
B_temp = zeros(n,n,n_classes);

for i=0:(n-1)
        for j = max(ceil(i-1-eps_state*mu),0):min(floor(i+1+eps_state*mu),n-1)
            B_temp(i+1,j+1,:) = (1/n).*P_temp(:,I{i+1,j+1}+1)*TransProbs{i+1,j+1}';
        end
        B_temp(i+1,i+1,:) = -sum(B_temp(i+1,:,:),2)+B_temp(i+1,i+1,:);
end
    
for k =1:n_classes
    B = zeros(n,n);
    for l =1:n_classes
        B = B + coeffs(k,l).*B_temp(:,:,l);
    end
    dydt1(((k-1)*n+1):((k)*n)) = P_temp(k,:)*B;
end
dydt = dydt1';