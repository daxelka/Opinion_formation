%%
load('../Pfinal_delta.mat')
% 
deltas = linspace(1,10,1000);

% Reconstrac corrected num # clusters
clusters_Num = ones(size(deltas));
biffurcations_corrected = [1.8600, 2.725, 4.0600, 4.93, 6.2300, 7.11, 8.3900, 9.28];

for i = 1:length(biffurcations_corrected)-1
    sampling = deltas >= biffurcations_corrected(i) & deltas < biffurcations_corrected(i+1);
    clusters_Num(sampling) = (i+1) * ones(size(clusters_Num(sampling)));
end

clusters_Num(deltas >= biffurcations_corrected(end)) = 9 * ones(size(clusters_Num(deltas >= biffurcations_corrected(end))));

% 1/2e rule
k = 1;
for i=2:9
    clusters_indecies(k) = find(floor(deltas) == i,1);
    k = k+1;
end
bif_points_BN = deltas(clusters_indecies);

% LS approximate rule
n_clusters_LS_appr = round(0.89 * deltas + 0.11);

% Biffurcation poitns
k = 1;
for i=2:9
    clusters_indecies(k) = find(n_clusters_LS_appr == i,1);
    k = k+1;
end
bif_points_LS_appr = deltas(clusters_indecies);



deltas_discrete = [2:1:9];
figure('Position', [0 0 550 400])
p_Num = plot(deltas, clusters_Num, 'b.', 'MarkerSize',12);
hold on

p_BN = scatter(bif_points_BN, [2:1:9],  60, 'k','o', 'filled','LineWidth',1.5);
hold on

for i = [1:1:8]
    plot([bif_points_BN(i) bif_points_BN(i)], [deltas_discrete(i)-0.5 deltas_discrete(i)+0.5],'k')
    hold on
end

p_LS = scatter(bif_points_LS_appr, [2:1:9], 60,'r','o','filled', 'LineWidth',1.5);
hold on

for i = [1:1:8]
    plot([bif_points_LS_appr(i) bif_points_LS_appr(i)], [deltas_discrete(i)-0.5 deltas_discrete(i)+0.5],'r')
    hold on
end

p_blueline = plot([1 1.2], [1 1], 'b', 'LineWidth', 2);
hold off


xlabel('\Delta')
ylabel('# clusters')
xlim([1,10])
ylim([0,9.5])
legend([p_blueline(1) p_LS(1) p_BN(1)],'Numerical soln.','Lin. stability approx. ','1/2\epsilon rule ',...
        'Location', 'southeast',...
        'Orientation','vertical',...
        'Box','off') 
set(gca,'Fontsize', 20)

