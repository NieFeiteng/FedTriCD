classdef ResultManager < handle
    properties
        result_file
        fid
        results_buffer
        header_written
    end

    methods
        function obj = ResultManager(filename)
            output_dir = fileparts(filename);
            if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
                mkdir(output_dir);
            end

            obj.result_file = filename;
            obj.results_buffer = {};
            obj.header_written = false;
            obj.fid = fopen(filename, 'a');

            if obj.fid == -1
                error('Could not open result file: %s', filename);
            end

            fseek(obj.fid, 0, 'eof');
            if ftell(obj.fid) == 0
                obj.writeHeader();
            else
                obj.header_written = true;
            end
        end

        function delete(obj)
            if ~isempty(obj.fid) && obj.fid ~= -1
                fclose(obj.fid);
                obj.fid = -1;
            end
        end

        function writeHeader(obj)
            if obj.header_written
                return;
            end

            fprintf(obj.fid, 'method_name,dataset_name,seed,n_points,n_clusters,avg_time,dist_count,iterations,sse,acc,nmi,ari,ami\n');
            obj.header_written = true;
        end

        function saveResult(obj, result_struct)
            fprintf(obj.fid, '%s,%s,%d,%d,%d,%.8f,%.2f,%.2f,%.12g,%.6f,%.6f,%.6f,%.6f\n', ...
                result_struct.method_name, ...
                result_struct.dataset_name, ...
                result_struct.seed, ...
                result_struct.n_points, ...
                result_struct.n_clusters, ...
                result_struct.avg_time, ...
                result_struct.avg_dist_count, ...
                result_struct.avg_iterations, ...
                result_struct.avg_sse, ...
                result_struct.acc, ...
                result_struct.nmi, ...
                result_struct.ari, ...
                result_struct.ami);

            obj.results_buffer{end + 1} = result_struct;
        end

        function summary = generateSummary(obj)
            n_results = numel(obj.results_buffer);
            fprintf('\n========== Experiment Summary ==========\n');
            fprintf('Results: %d\n', n_results);
            fprintf('File: %s\n', obj.result_file);
            fprintf('========================================\n\n');
            summary = sprintf('Finished %d experiments.', n_results);
        end
    end
end
