
peak_canonical = [1.26,3.44,5.61,7.74,9.89];

deltas = linspace(1,10,500);


deltas_sym=cell(4,1);
for delta = deltas
    % cut the domain 
    domain_lenght = delta;

    % pinpoint symmetric peaks
    peaks_in_domain = peak_canonical(peak_canonical<=domain_lenght-0.5);

    if ~isempty(peaks_in_domain)
        for i = 1:length(peaks_in_domain)
            % transform to local delta grid
            peaks_local = peaks_in_domain(i) - delta;
        
            % symmetric clusters
            deltas_sym{i} =  [deltas_sym{i}; i];
%             peaks_sym(i) = peaks_in_domain(i)- delta;
        end
    end

%     % pinpoint central peaks
%     peaks_in_centre = peak_canonical((peak_canonical>domain_lenght-0.5)&(peak_canonical<domain_lenght));
%     peaks_central_local = peaks_in_centre -delta;
%     deltas_central = delta*ones(size(peaks_in_centre));
% 
%     peaks_central_all = [peaks_central_all peaks_central_local];
%     deltas_central_all = [deltas_central_all deltas_central];
% 
%    
%     % move central peaks to a centre
%     if ~isempty(peaks_in_centre)
%         delta_central_true = [delta_central_true delta];
%     end
end
% % preparing results
% peaks_sym_sorted = sortrows([deltas_sym', peaks_sym'],1);
% y_diff = diff(peaks_sym_sorted(:,2));
% 
% % end
% 
% % plot(deltas_sym, peaks_sym, 'b.')
% plot(peaks_sym_sorted(:,1), peaks_sym_sorted(:,2), 'b.')
% hold on
% plot(deltas_central_all, peaks_central_all,'r.')
% hold on
% 
% plot(delta_central_true, zeros(size(delta_central_true)), 'Color', 'r','LineStyle','none', 'Marker','.');
% 
% xlabel('\Delta')
% ylabel('y')
% set(gca,'Fontsize', 20)
% 
% hold off
