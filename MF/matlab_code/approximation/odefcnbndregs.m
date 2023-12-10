function dydt = odefcnbndregs(t,P, degrees, degree_dist )

n = length(P)/length(degrees);
J = n-1;
avg_degree = degrees*degree_dist.';

P_temp = reshape(P,n,length(degrees))'; %length(degrees)X n matrix
P_temp = [P_temp,ones(length(degrees),n)]; %Fix P=1 for x>1
%J = length(P)-1;%P here is just defined on the boundary region

dydt=zeros(length(P),1);
dydt_temp = zeros(length(degrees),n);

for k = 1:length(degrees)
    for l=1:length(degrees)
        for j=0:J
            %If probability between (x_j - eps/2,x_j + eps/2) is zero, probability
            %at x_j doesnt change
            %region = max(0,(j-epsind1)):min(J,(j+epsind1));
            %if sum(P(region+1)>threshold)==0
            %    dydt(j+1)=0;
            %else
    
            %Integral 1
            liml1 = 0;%max([0,j-epsind1,2*j-J]);
            limu1 = min(j,J/2);%min([J,j+epsind1,2*j]);
            if liml1==limu1
                I1 =0;
            else
                x1 = liml1:1:limu1;
                y1 = P_temp(k,j+x1+1).*P_temp(l,j-x1+1);
                I1 = trapz(y1)/J;
            end

            %Integral 2
            liml2 = 0;%max([0,j-epsind2]);
            limu2 = J+j;%min([J,j+epsind2]);
            x2 = liml2:1:limu2;
            y2 = P_temp(l,x2+1);
            I2 = P_temp(k,j+1)*trapz(y2)/J;

            dydt_temp(k,j+1) = dydt_temp(k,j+1)+(degrees(k).*(degrees(l).*degree_dist(l)/(avg_degree^2)))*(4*I1-I2);
        end
    end
end
for k = 1:length(degrees)
    dydt(((k-1)*n+1):((k)*n)) = dydt_temp(k,:);
end
