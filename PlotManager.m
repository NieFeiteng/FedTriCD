classdef PlotManager
    methods (Static)
        function plotConvergence(sse_data, method_labels, config_info, save_path)
            markers = {'-o', '-s', '-^', '-d', '-*', '-v', '->', '-<', '-p', '-h', '-+', '-x'};
            colors = lines(numel(method_labels));
            [num_methods, T] = size(sse_data);
            x = 1:T;
            sse_plot = PlotManager.handleInvalidTail(sse_data);

            figure('Position', [100, 100, 800, 600], 'Visible', 'off');
            hold on;

            for i = 1:num_methods
                marker_style = markers{min(i, numel(markers))};
                plot(x, sse_plot(i, :), marker_style, 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', colors(i, :), 'DisplayName', method_labels{i});
            end

            xlabel('Iteration', 'FontSize', 12);
            ylabel('Objective (SSE)', 'FontSize', 12);
            title(sprintf('Dataset: %s | n=%d | k=%d', config_info.dataset_name, config_info.n_points, config_info.n_clusters), 'Interpreter', 'none');
            legend('Location', 'northeast', 'FontSize', 10);
            grid on;
            hold off;
            saveas(gcf, save_path);
            close(gcf);

            fprintf('Saved plot: %s\n', save_path);
        end

        function sse_plot = handleInvalidTail(sse_data)
            [num_methods, T] = size(sse_data);
            sse_plot = sse_data;

            for i = 1:num_methods
                row = sse_plot(i, :);
                last_valid = find(isfinite(row), 1, 'last');

                if isempty(last_valid)
                    row(:) = NaN;
                elseif last_valid < T
                    row(last_valid + 1:end) = NaN;
                end

                sse_plot(i, :) = row;
            end
        end

        function ensureOutputDir(output_dir)
            if exist(output_dir, 'dir') ~= 7
                mkdir(output_dir);
            end
        end

        function filename = generatePlotFilename(config_info)
            filename = sprintf('Data_%s_n%d_C%d_Seed%d.png', config_info.dataset_name, config_info.n_points, config_info.n_clusters, config_info.seed);
        end
    end
end
