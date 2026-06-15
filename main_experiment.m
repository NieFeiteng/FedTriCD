function main_experiment()
root_dir = fileparts(mfilename('fullpath'));
methods_dir = fullfile(root_dir, 'methods');

addpath(root_dir);
if exist(methods_dir, 'dir') ~= 7
    error('Methods directory not found: %s', methods_dir);
end
addpath(methods_dir);

clc;
close all;

fprintf('Starting CDKM, LLoyd, and TriCD experiments.\n\n');

config = createExperimentConfig(root_dir);
registry = registerCoreMethods();

runner = ExperimentRunner(config);
runner.method_registry = registry;

try
    runner.runAllExperiments();
catch ME
    fprintf('Error: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('%s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    rethrow(ME);
end
end

function config = createExperimentConfig(root_dir)
config = ExperimentConfig(root_dir);
end

function registry = registerCoreMethods()
registry = MethodRegistry();
registry = registry.registerMethod('CDKM', @CDKM);
registry = registry.registerMethod('LLoyd', @LLoyd);
registry = registry.registerMethod('TriCD', @TriCD);
end
