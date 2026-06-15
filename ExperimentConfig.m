classdef ExperimentConfig
    properties
        project_root = ''
        methods_to_test = {'CDKM', 'LLoyd', 'TriCD'}
        datasets = {'YearPredictionMSD', 'Epileptic', 'NYC', 'kegg', 'Crime', 'shuttle', 'cifar10', 'cifar100', 'letter', 'mnist'}
        data_path = ''
        max_points = 10000
        feature_dim_limit = 5000
        cluster_nums = 4
        max_iterations = 200
        num_runs = 1
        seeds = 2
        normalization_type = 'none'
        result_file = ''
        output_picture_dir = ''
        enable_plotting = true
    end

    methods
        function obj = ExperimentConfig(project_root)
            if nargin < 1 || isempty(project_root)
                project_root = pwd;
            end

            obj.project_root = project_root;

            env_data_path = getenv('CLUSTER_DATA_PATH');
            if isempty(env_data_path)
                obj.data_path = fullfile(project_root, 'data');
            else
                obj.data_path = env_data_path;
            end

            obj.result_file = fullfile(project_root, 'results', 'experiment_results.csv');
            obj.output_picture_dir = fullfile(project_root, 'results', 'plots');
        end

        function validate(obj)
            allowed_methods = {'CDKM', 'LLoyd', 'TriCD'};
            unknown_methods = setdiff(obj.methods_to_test, allowed_methods);

            assert(isempty(unknown_methods), 'Unknown method: %s', strjoin(unknown_methods, ', '));
            assert(~isempty(obj.methods_to_test), 'At least one method is required.');
            assert(~isempty(obj.datasets), 'At least one dataset is required.');
            assert(all(obj.cluster_nums > 0), 'cluster_nums must be positive.');
            assert(all(obj.max_points > 0), 'max_points must be positive.');
            assert(obj.max_iterations > 0, 'max_iterations must be positive.');
            assert(obj.num_runs > 0, 'num_runs must be positive.');
            assert(all(obj.seeds >= 0), 'seeds must be nonnegative.');
        end

        function displayConfig(obj)
            fprintf('========== Experiment Config ==========\n');
            fprintf('Methods: %s\n', strjoin(obj.methods_to_test, ', '));
            fprintf('Datasets: %s\n', strjoin(obj.datasets, ', '));
            fprintf('Data path: %s\n', obj.data_path);
            fprintf('Cluster nums: %s\n', mat2str(obj.cluster_nums));
            fprintf('Max points: %s\n', mat2str(obj.max_points));
            fprintf('Max iterations: %d\n', obj.max_iterations);
            fprintf('Runs: %d\n', obj.num_runs);
            fprintf('Seeds: %s\n', mat2str(obj.seeds));
            fprintf('Result file: %s\n', obj.result_file);
            fprintf('=======================================\n\n');
        end
    end
end
