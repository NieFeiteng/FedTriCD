function [Y, minO, iter_num, obj, elapsed_time, update_record, dist_count] = LLoyd(X, label, c, iter_rounds)
fprintf('LLoyd\n');

label = label(:);

obj = zeros(1, iter_rounds + 1);
update_record = zeros(1, iter_rounds);
dist_count = 0;
iter_num = 0;

centers = computeCenters(X, label, c, []);
obj(1) = computeSSE(X, label, centers);

run_time = tic;

for iter = 1:iter_rounds
    [new_label, iter_dist_count] = assignLabels(X, centers);
    dist_count = dist_count + iter_dist_count;
    update_record(iter) = sum(new_label ~= label);
    label = new_label;
    centers = computeCenters(X, label, c, centers);
    obj(iter + 1) = computeSSE(X, label, centers);
    iter_num = iter;

    if abs(obj(iter + 1) - obj(iter)) < 1e-5 || update_record(iter) == 0
        obj(iter + 2:end) = obj(iter + 1);
        break;
    end
end

elapsed_time = toc(run_time);
minO = min(obj);
Y = label;

fprintf('Total changes: %d\n', sum(update_record));
fprintf('Elapsed time: %.4f seconds\n', elapsed_time);
end

function centers = computeCenters(X, label, c, previous_centers)
[d, n] = size(X);
centers = zeros(d, c);

for k = 1:c
    idx = label == k;

    if any(idx)
        centers(:, k) = mean(X(:, idx), 2);
    elseif isempty(previous_centers)
        centers(:, k) = X(:, mod(k - 1, n) + 1);
    else
        centers(:, k) = previous_centers(:, k);
    end
end
end

function [label, dist_count] = assignLabels(X, centers)
[~, n] = size(X);
c = size(centers, 2);
distances = zeros(c, n);

for k = 1:c
    diff = X - centers(:, k);
    distances(k, :) = sum(diff.^2, 1);
end

[~, label] = min(distances, [], 1);
label = label(:);
dist_count = n * c;
end

function sse = computeSSE(X, label, centers)
sse = 0;
c = size(centers, 2);

for k = 1:c
    idx = label == k;
    if any(idx)
        diff = X(:, idx) - centers(:, k);
        sse = sse + sum(sum(diff.^2, 1));
    end
end
end
