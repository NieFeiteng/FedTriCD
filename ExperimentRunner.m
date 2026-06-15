classdef ExperimentRunner < handle
    properties
        config
        method_registry
        result_manager
        data_loader
    end

    methods
        function obj = ExperimentRunner(config)
            obj.config = config;
            obj.config.validate();
            obj.result_manager = ResultManager(config.result_file);
            obj.data_loader = DataLoader();
            PlotManager.ensureOutputDir(config.output_picture_dir);
        end

        function delete(obj)
            if ~isempty(obj.result_manager)
                delete(obj.result_manager);
            end
        end

        function runAllExperiments(obj)
            obj.config.displayConfig();

            total_experiments = numel(obj.config.seeds) * numel(obj.config.datasets) * numel(obj.config.max_points) * numel(obj.config.cluster_nums) * numel(obj.config.methods_to_test);
            fprintf('Total experiments: %d\n\n', total_experiments);

            exp_count = 0;
            start_time = tic;

            for seed_idx = 1:numel(obj.config.seeds)
                seed = obj.config.seeds(seed_idx);

                for dataset_idx = 1:numel(obj.config.datasets)
                    dataset_name = obj.config.datasets{dataset_idx};

                    try
                        [X, metadata] = obj.data_loader.loadDataset(dataset_name, obj.config);
                    catch ME
                        fprintf('Skipping dataset %s: %s\n', dataset_name, ME.message);
                        continue;
                    end

                    for point_idx = 1:numel(obj.config.max_points)
                        max_points = obj.config.max_points(point_idx);
                        [X_processed, metadata] = obj.data_loader.preprocessData(X, metadata, obj.config, max_points);
                        n_points = size(X_processed, 2);

                        for cluster_idx = 1:numel(obj.config.cluster_nums)
                            n_clusters = obj.config.cluster_nums(cluster_idx);
                            sse_matrix = NaN(numel(obj.config.methods_to_test), obj.config.max_iterations + 1);

                            for method_idx = 1:numel(obj.config.methods_to_test)
                                method_name = obj.config.methods_to_test{method_idx};
                                exp_count = exp_count + 1;

                                fprintf('[%d/%d] %s | %s | k=%d | n=%d\n', exp_count, total_experiments, method_name, dataset_name, n_clusters, n_points);

                                result = obj.runSingleConfiguration(X_processed, metadata, method_name, n_clusters, seed);
                                obj.result_manager.saveResult(result);
                                sse_matrix(method_idx, :) = result.sse_history;

                                fprintf('  time %.4fs | sse %.10g | iterations %.2f | distances %.2f\n', result.avg_time, result.avg_sse, result.avg_iterations, result.avg_dist_count);
                            end

                            if obj.config.enable_plotting
                                obj.generateComparisonPlot(sse_matrix, dataset_name, n_points, n_clusters, seed);
                            end
                        end
                    end
                end
            end

            total_time = toc(start_time);
            fprintf('\nAll experiments finished in %.4f seconds.\n', total_time);
            obj.result_manager.generateSummary();
        end

        function result = runSingleConfiguration(obj, X, metadata, method_name, n_clusters, seed)
            n_points = size(X, 2);
            total_time = 0;
            total_sse = 0;
            total_iterations = 0;
            total_dist_count = 0;
            sse_history_sum = zeros(1, obj.config.max_iterations + 1);

            for run_idx = 1:obj.config.num_runs
                run_seed = seed + run_idx - 1;
                label = obj.data_loader.initializeLabels(n_points, n_clusters, run_seed);

                try
                    [~, ~, iter_num, sse_history, elapsed_time, ~, dist_count] = obj.method_registry.runMethod(method_name, X, label, n_clusters, obj.config.max_iterations);
                    sse_history = obj.normalizeHistory(sse_history);
                    finite_sse = sse_history(isfinite(sse_history));

                    if isempty(finite_sse)
                        best_sse = inf;
                    else
                        best_sse = min(finite_sse);
                    end

                    total_time = total_time + elapsed_time;
                    total_sse = total_sse + best_sse;
                    total_iterations = total_iterations + iter_num;
                    total_dist_count = total_dist_count + dist_count;
                    sse_history_sum = sse_history_sum + sse_history;
                catch ME
                    fprintf('  failed: %s\n', ME.message);
                    total_sse = total_sse + inf;
                end
            end

            result.method_name = method_name;
            result.dataset_name = metadata.dataset_name;
            result.seed = seed;
            result.n_points = n_points;
            result.n_clusters = n_clusters;
            result.avg_time = total_time / obj.config.num_runs;
            result.avg_dist_count = total_dist_count / obj.config.num_runs;
            result.avg_iterations = total_iterations / obj.config.num_runs;
            result.avg_sse = total_sse / obj.config.num_runs;
            result.sse_history = sse_history_sum / obj.config.num_runs;
            result.acc = 0;
            result.nmi = 0;
            result.ari = 0;
            result.ami = 0;
        end

        function generateComparisonPlot(obj, sse_matrix, dataset_name, n_points, n_clusters, seed)
            config_info.dataset_name = dataset_name;
            config_info.n_points = n_points;
            config_info.n_clusters = n_clusters;
            config_info.seed = seed;
            filename = PlotManager.generatePlotFilename(config_info);
            save_path = fullfile(obj.config.output_picture_dir, filename);
            PlotManager.plotConvergence(sse_matrix, obj.config.methods_to_test, config_info, save_path);
        end

        function history = normalizeHistory(obj, history)
            target_len = obj.config.max_iterations + 1;
            history = history(:)';

            if isempty(history)
                history = NaN(1, target_len);
                return;
            end

            if numel(history) < target_len
                history(end + 1:target_len) = history(end);
            elseif numel(history) > target_len
                history = history(1:target_len);
            end
        end
    end
end
