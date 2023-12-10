function [mass,location] = clusterloc(P,threshold)

J = length(P)-1;
%posProb = P*(J+1) >threshold; %elements of P greater than cutoff
posProb = P >threshold; %elements of P greater than cutoff
diffx = diff(posProb); 
%Difference between sequential elements of posProb. diffx =1 if distribution
%goes from 0 to positive,-1 if it goes from positive to 0 
locTrans = find(diffx ~=0); %location of transitions from + to 0,0 to +
%transType = diffx(locTrans); %Type of transition
lenInts = diff([0,locTrans,length(P)]); % length of intervals
regionType = [(posProb(locTrans)),posProb(end)];
lenRegionNoProb = lenInts(regionType==0);
lenRegionPosProb = lenInts(regionType==1);

%Mass and location of clusters
mass = zeros(1,length(lenRegionPosProb));
location = zeros(1,length(lenRegionPosProb));
points = [0,locTrans,J+1];
k=0;
for j=1:length(regionType)
    if regionType(j)==0
        continue
    else          
        k=k+1;
        x = (points(j)+1):points(j+1);
        mass(k) = sum(P(x));
        location(k) = sum((x-1).*P(x)/J)/mass(k);
    end
end