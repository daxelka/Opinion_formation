function[value,isterminal,direction] = DeffuantEvents(t,P,threshold,eps_state,degree_dist)
%Want the solver to stop if the distribution has separated into
%non-interacting clusters (clusters separated by epsilon with with less
%than epsiln)
%Need the probability density to have cutoff, i.e., P*n < threshold to be
%treated as zero. 
dist_error = 0.01;
n_degrees = length(degree_dist);
n = length(P)/n_degrees;
P_temp = zeros(n_degrees,n);
for i = 1:n_degrees
    P_temp(i,:) = degree_dist(i).*P((((i-1)*n)+1):(n*(i)));
end
P_temp = sum(P_temp,1);
if (sum(P_temp(P_temp<threshold))/n)>dist_error 
    %If the area in the regions with "no probability" (P less than
    %threshold) is bigger than dist_error then we say that we haven't converged
    value = [1;1]; 
else
    posProb = P_temp >threshold; %elements of P greater than cutoff
    diffx = diff(posProb); 
    %Difference between sequential elements of posProb. diffx =1 if distribution
    %goes from 0 to positive,-1 if it goes from positive to 0 
    locTrans = find(diffx ~=0); %location of transitions from + to 0,0 to +
    %transType = diffx(locTrans); %Type of transition
    lenInts = diff([0,locTrans,length(P_temp)]); % length of intervals
    regionType = [(posProb(locTrans)),posProb(end)];
    lenRegionNoProb = lenInts(regionType==0);
    lenRegionPosProb = lenInts(regionType==1);

    if length(lenRegionPosProb)==1
        %If there is one cluster, check the cluster width is less than
        %epsilon
        regionsSep = (lenRegionPosProb<eps_state);
    else
        %If there are multiple clusters, check each cluster width is less
        %than epsilon and the distance between clusters is greater than
        %epsilon
        regionsSep = (sum([lenRegionNoProb<(eps_state-1),lenRegionPosProb>=eps_state])==0);
    end
    %Want to make sure we aren't getting large numerical errors (can happen
    %if some areas of the distribution converge to delta spikes while other
    %areas are still flat - this happens with a large number of
    %clusters i.e., small values of epsilon)
    totProb = sum(P_temp)/n;

    value = [1-regionsSep;1*(totProb < 1.01)];
end
isterminal = [1;1];
direction = [0;0];