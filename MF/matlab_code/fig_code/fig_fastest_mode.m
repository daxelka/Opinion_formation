%% Plot of s as a function of x = eps*k
%figure
x = linspace(0,5,1000);
y = s_over_2eps(x);
figure('Position', [0 0 400 400])
plot(x,y,'r', 'LineWidth', 1.5)
ylim([-0.4 0.4])
xlim([0,5])
ylabel('$$\tilde{s}$$', 'Interpreter', 'LaTeX')
xlabel('$$\tilde{k}$$', 'Interpreter', 'LaTeX')
set(gca,'Fontsize', 20)

% k_tilde_star which maximises s_tilde
[val, idx] = max(y);
x(idx)

% Functions
function[s_val] = s_over_2eps(x)
    s_val = zeros(size(x));
    x_temp = x(x~=0);
    s_val(x~=0) = (1./x_temp).*(4.*sin(x_temp./2)-sin(x_temp))-1;
end
