classdef MethodRegistry
    properties (Access = private)
        methods_map
    end

    methods
        function obj = MethodRegistry()
            obj.methods_map = containers.Map();
        end

        function obj = registerMethod(obj, name, func_handle)
            method_info.name = name;
            method_info.func_handle = func_handle;
            obj.methods_map(name) = method_info;
        end

        function [Y_label, best_objective, iter_num, sse_history, elapsed_time, extra_info, dist_count] = runMethod(obj, method_name, X, label, c, max_iter)
            if ~obj.methods_map.isKey(method_name)
                error('Method is not registered: %s', method_name);
            end

            method_info = obj.methods_map(method_name);
            func = method_info.func_handle;
            [Y_label, best_objective, iter_num, sse_history, elapsed_time, extra_info, dist_count] = func(X, label, c, max_iter);
        end

        function names = getMethodNames(obj)
            names = obj.methods_map.keys();
        end

        function exists = hasMethod(obj, method_name)
            exists = obj.methods_map.isKey(method_name);
        end
    end
end
