classdef DataLoader
    methods (Static)
        function [X, metadata] = loadDataset(dataset_name, config)
            csv_path = fullfile(config.data_path, [dataset_name, '.csv']);
            txt_path = fullfile(config.data_path, [dataset_name, '.txt']);

            if exist(csv_path, 'file') == 2
                data = DataLoader.readCsv(csv_path);
                X = data';
                file_path = csv_path;
                is_sparse = false;
            elseif exist(txt_path, 'file') == 2
                data = load(txt_path);
                X = spconvert(data)';
                file_path = txt_path;
                is_sparse = true;
            else
                error('Dataset file not found: %s.csv or %s.txt', dataset_name, dataset_name);
            end

            metadata.dataset_name = dataset_name;
            metadata.original_size = size(X);
            metadata.file_path = file_path;
            metadata.is_sparse = is_sparse;
            metadata.feature_dim = min(config.feature_dim_limit, size(X, 1));

            fprintf('Loaded %s: %d x %d\n', dataset_name, size(X, 1), size(X, 2));
        end

        function [X, metadata] = preprocessData(X, metadata, config, max_points)
            n_original = size(X, 2);
            n_samples = min(max_points, n_original);
            d_limit = min(metadata.feature_dim, size(X, 1));

            X = X(1:d_limit, 1:n_samples);
            metadata.actual_size = size(X);
            metadata.sample_ratio = n_samples / n_original;

            switch lower(config.normalization_type)
                case 'zscore'
                    X = normalize(X, 2);
                case 'range'
                    X = normalize(X, 2, 'range');
                case 'none'
                otherwise
                    error('Unknown normalization type: %s', config.normalization_type);
            end

            fprintf('Prepared %s: %d x %d, sample %.2f%%\n', metadata.dataset_name, size(X, 1), size(X, 2), metadata.sample_ratio * 100);
        end

        function label = initializeLabels(n_points, n_clusters, seed)
            rng(seed, 'twister');
            label = randi(n_clusters, n_points, 1);
        end

        function data = readCsv(file_path)
            data = readmatrix(file_path);
            data = data(~all(isnan(data), 2), :);
            data = data(:, ~all(isnan(data), 1));
        end
    end
end
