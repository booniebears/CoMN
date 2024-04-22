% Figure 1 in the rbf_rt_2 manual page.
clear

% Hermite training data.
[x, y] = get_data('hermite');

% Test data.
test.name = 'hermite';
test.p = 1000;
test.ord = 1;
test.std = 0;
[xt, yt] = get_data(test);

% Run the method with the default configuration.
[c, r, w, info] = rbf_rt_2(x, y);

% Now do the test set predictions.
Ht = rbf_dm(xt, c, r, info.dmc);
ft = Ht * w;

% Get figure.
fig = get_fig('Figure 1');

% Plot.
hold off
plot(xt, yt, 'k--')
hold on
plot(xt, ft, 'r-', 'LineWidth', 2)

% Configure plot.
set(gca, 'FontSize', 16)
set(gca, 'Position', [0.1 0.15 0.85 0.8])
set(gca, 'XLim', [-4 4])
set(gca, 'YLim', [0 3])
set(gca, 'XTick', -4:2:4)
set(gca, 'YTick', 0:3)
xlabel('x', 'FontSize', 16)
ylabel('y', 'FontSize', 16, 'Rotation', 0)
legend('target', 'prediction')

% Save postscript.
print -depsc fig1
