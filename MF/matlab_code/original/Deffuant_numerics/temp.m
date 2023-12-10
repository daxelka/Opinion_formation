for i=1:size(Ptot,1)
    Px(i,:) = diff(Ptot(i,:))./diff(linspace(0,1,666));
end

for i=1:size(Ptot,1)
    Pxx(i,:) = diff(Px(i,:))./diff(linspace(0,1,665));
end

% plot(linspace(0,1,665), Px(11,:))
index = 11;
figure()
plot(linspace(0,1,665), Px(index,:), linspace(0,1,665), Ptot(index,2:end))
% ylim([0 5])

figure()
plot(linspace(0,1,664), Pxx(index,:), linspace(0,1,665), Ptot(index,2:end))