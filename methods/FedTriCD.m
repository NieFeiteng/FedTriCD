function [Y, minO, iter_num, obj, elapsed_time, update_record, total_computed] = FedTriCD(X, label, c, max_iters, seed, clients_num, lambda_prox)
if nargin < 5 || isempty(seed)
    seed = 1;
end
if nargin < 6 || isempty(clients_num)
    clients_num = 4;
end
if nargin < 7 || isempty(lambda_prox)
    lambda_prox = 0.01;
end

label = label(:);
[d, n] = size(X);

if numel(label) ~= n
    error('label must contain one entry per data point.');
end
if any(label < 1) || any(label > c)
    error('label values must be in 1..c.');
end
if clients_num < 1 || clients_num > d
    error('clients_num must be between 1 and the feature dimension.');
end

local_rounds = 2;
tol = 1e-5;
patience = 10;

feature_blocks = splitFeatures(d, clients_num, seed);
counts = clusterCounts(label, c);
clients = initClients(X, label, counts, feature_blocks, c);

obj = zeros(1, max_iters + 1);
obj(1) = serverSSE(X, label, c);
update_record = zeros(max_iters, 1);
total_computed = 0;
iter_num = 0;

best_obj = obj(1);
best_label = label;
no_improve = 0;

run_time = tic;

for iter = 1:max_iters
    phi = zeros(n, c, clients_num, 'single');

    for local_iter = 1:local_rounds
        for client_id = 1:clients_num
            [clients(client_id), phi(:, :, client_id), computed] = updateClient(clients(client_id), c, lambda_prox);
            total_computed = total_computed + computed;
        end
    end

    [label, counts, moved] = updateServer(label, counts, sum(phi, 3));
    clients = syncClients(clients, label, counts, c);

    update_record(iter) = moved;
    obj(iter + 1) = serverSSE(X, label, c);
    iter_num = iter;

    improvement = (best_obj - obj(iter + 1)) / max(abs(best_obj), 1);
    if improvement > tol
        best_obj = obj(iter + 1);
        best_label = label;
        no_improve = 0;
    else
        no_improve = no_improve + 1;
    end

    if moved == 0 || no_improve >= patience
        label = best_label;
        obj(iter + 2:end) = best_obj;
        break;
    end
end

elapsed_time = toc(run_time);
Y = label;
minO = min(obj);
update_record = update_record(1:iter_num);
end

function blocks = splitFeatures(d, clients_num, seed)
rng(seed, 'twister');
order = randperm(d);
base_size = floor(d / clients_num);
extra = mod(d, clients_num);
blocks = cell(1, clients_num);
start_idx = 1;

for client_id = 1:clients_num
    block_size = base_size + (client_id <= extra);
    end_idx = start_idx + block_size - 1;
    blocks{client_id} = sort(order(start_idx:end_idx));
    start_idx = end_idx + 1;
end
end

function counts = clusterCounts(label, c)
counts = zeros(1, c);

for k = 1:c
    counts(k) = sum(label == k);
end
end

function clients = initClients(X, label, counts, feature_blocks, c)
template = struct('X', [], 'XX', [], 'norms', [], 'prox', [], 'label', [], 'anchor_label', [], 'counts', [], 'XmF', [], 'FXXF', [], 'lb', [], 'ub', []);
clients = repmat(template, 1, numel(feature_blocks));

for client_id = 1:numel(feature_blocks)
    Xm = X(feature_blocks{client_id}, :);
    client.X = Xm;
    client.XX = sum(Xm.^2, 1);
    client.norms = sqrt(client.XX);
    client.prox = zeros(c, 1);
    client = syncClient(client, label, counts, c);
    clients(client_id) = client;
end
end

function clients = syncClients(clients, label, counts, c)
for client_id = 1:numel(clients)
    clients(client_id) = syncClient(clients(client_id), label, counts, c);
end
end

function client = syncClient(client, label, counts, c)
[feature_dim, ~] = size(client.X);
client.label = label(:);
client.anchor_label = client.label;
client.counts = counts;
client.XmF = zeros(feature_dim, c);

for k = 1:c
    idx = client.label == k;
    if any(idx)
        client.XmF(:, k) = sum(client.X(:, idx), 2);
    end
end

client.FXXF = sum(client.XmF.^2, 1);
xB = client.X' * client.XmF;
client.lb = xB;
client.ub = xB;
client.prox(:) = 0;
end

function [client, phi, computed] = updateClient(client, c, lambda_prox)
tol = 1e-10;
[feature_dim, n] = size(client.X);
phi = zeros(n, c, 'single');
computed = 0;
cluster_dist = clusterDistances(client.XmF, client.counts, c);
delta_BB = zeros(feature_dim, c);
dBB_norms = zeros(1, c);

for i = 1:n
    current_cluster = client.label(i);
    current_count = client.counts(current_cluster);

    if current_count <= 1
        continue;
    end

    xi = client.X(:, i);
    xi_norm = client.norms(i);
    xi_sq = client.XX(i);
    x_current = xi' * client.XmF(:, current_cluster);
    v_remove = client.FXXF(current_cluster) - 2 * x_current + xi_sq;
    current_cost = v_remove / (current_count - 1) - client.FXXF(current_cluster) / current_count;
    current_phi = current_cost + 0.5 * lambda_prox * client.prox(current_cluster);
    computed = computed + 1;

    phi(i, current_cluster) = current_phi;
    client.lb(i, current_cluster) = x_current;
    client.ub(i, current_cluster) = x_current;
    best_cluster = current_cluster;
    best_phi = current_phi;

    for k = 1:c
        if k == current_cluster
            continue;
        end

        target_count = client.counts(k);
        if target_count == 0
            phi(i, k) = realmax('single');
            continue;
        end

        safe_distance = cluster_dist(current_cluster, k) + dBB_norms(current_cluster) + dBB_norms(k);
        lower_bound = x_current - xi_norm * safe_distance;
        upper_bound = x_current + xi_norm * safe_distance;
        old_lower = client.lb(i, k);
        old_upper = client.ub(i, k);
        tight_lower = max(old_lower, lower_bound);
        tight_upper = min(old_upper, upper_bound);

        if tight_lower > tight_upper + tol
            tight_lower = old_lower;
            tight_upper = old_upper;
        end

        client.lb(i, k) = tight_lower;
        client.ub(i, k) = tight_upper;
        effective_lower = tight_lower - xi_norm * dBB_norms(k);
        effective_upper = tight_upper + xi_norm * dBB_norms(k);
        v_add_upper = client.FXXF(k) + 2 * effective_upper + xi_sq;
        target_lower_cost = client.FXXF(k) / target_count - v_add_upper / (target_count + 1);
        target_lower_phi = target_lower_cost + 0.5 * lambda_prox * client.prox(k);

        if target_lower_phi > best_phi
            v_add_lower = client.FXXF(k) + 2 * effective_lower + xi_sq;
            target_upper_cost = client.FXXF(k) / target_count - v_add_lower / (target_count + 1);
            phi(i, k) = target_upper_cost + 0.5 * lambda_prox * client.prox(k);
            continue;
        end

        x_target = xi' * client.XmF(:, k);
        v_add = client.FXXF(k) + 2 * x_target + xi_sq;
        target_cost = client.FXXF(k) / target_count - v_add / (target_count + 1);
        target_phi = target_cost + 0.5 * lambda_prox * client.prox(k);
        computed = computed + 1;

        client.lb(i, k) = x_target;
        client.ub(i, k) = x_target;
        phi(i, k) = target_phi;

        if target_phi < best_phi
            best_phi = target_phi;
            best_cluster = k;
        end
    end

    if best_cluster ~= current_cluster
        client = moveLocalPoint(client, i, xi, current_cluster, best_cluster, v_remove);
        delta_BB(:, best_cluster) = delta_BB(:, best_cluster) + xi;
        delta_BB(:, current_cluster) = delta_BB(:, current_cluster) - xi;
        dBB_norms(best_cluster) = norm(delta_BB(:, best_cluster));
        dBB_norms(current_cluster) = norm(delta_BB(:, current_cluster));
    end
end

for k = 1:c
    if dBB_norms(k) > 1e-12
        shift = client.norms * dBB_norms(k);
        client.lb(:, k) = client.lb(:, k) - shift';
        client.ub(:, k) = client.ub(:, k) + shift';
    end
end
end

function client = moveLocalPoint(client, i, xi, old_cluster, new_cluster, old_cluster_norm)
client.XmF(:, new_cluster) = client.XmF(:, new_cluster) + xi;
client.XmF(:, old_cluster) = client.XmF(:, old_cluster) - xi;
client.counts(new_cluster) = client.counts(new_cluster) + 1;
client.counts(old_cluster) = client.counts(old_cluster) - 1;
client.FXXF(old_cluster) = old_cluster_norm;
client.FXXF(new_cluster) = sum(client.XmF(:, new_cluster).^2);

if new_cluster == client.anchor_label(i)
    client.prox(new_cluster) = client.prox(new_cluster) - 1;
else
    client.prox(new_cluster) = client.prox(new_cluster) + 1;
end

scale = max(max(abs(client.prox)), 1);
client.prox = client.prox / scale;
client.label(i) = new_cluster;
client.lb(i, new_cluster) = xi' * client.XmF(:, new_cluster);
client.ub(i, new_cluster) = client.lb(i, new_cluster);

if client.counts(old_cluster) > 0
    client.lb(i, old_cluster) = xi' * client.XmF(:, old_cluster);
    client.ub(i, old_cluster) = client.lb(i, old_cluster);
end
end

function distances = clusterDistances(XmF, counts, c)
distances = zeros(c, c);
active = find(counts > 0);

for left_idx = 1:numel(active)
    left = active(left_idx);
    for right_idx = left_idx + 1:numel(active)
        right = active(right_idx);
        distance = norm(XmF(:, left) - XmF(:, right));
        distances(left, right) = distance;
        distances(right, left) = distance;
    end
end
end

function [label, counts, moved] = updateServer(label, counts, phi)
moved = 0;
n = numel(label);

for i = 1:n
    old_cluster = label(i);

    if counts(old_cluster) <= 1
        continue;
    end

    [~, new_cluster] = min(phi(i, :));

    if new_cluster ~= old_cluster
        label(i) = new_cluster;
        counts(new_cluster) = counts(new_cluster) + 1;
        counts(old_cluster) = counts(old_cluster) - 1;
        moved = moved + 1;
    end
end
end

function value = serverSSE(X, label, c)
value = 0;

for k = 1:c
    idx = label == k;
    if any(idx)
        Xk = X(:, idx);
        center = mean(Xk, 2);
        diff = Xk - center;
        value = value + sum(sum(diff.^2, 1));
    end
end
end
