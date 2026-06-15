function [Y, minO, iter_num, obj, elapsed_time, update_record, dist_count] = TriCD(X, label, c, iter_rounds)
fprintf('TriCD\n');

label = label(:);
[d, n] = size(X);

XX = sum(X.^2, 1);
xi_norms = sqrt(XX);
total_xx = sum(XX);
inv_n_table = [0, 1 ./ (1:n)];

[BB, aa, FXXF] = buildStats(X, label, c);

raw_products = X' * BB;
ub_xB = raw_products';

obj = zeros(1, iter_rounds + 1);
obj(1) = currentObjective(total_xx, FXXF, aa);
update_record = zeros(1, iter_rounds);
iter_num = 0;
dist_count = 0;
change_count = 0;

run_time = tic;

for iter = 1:iter_rounds
    total_tests = 0;
    pruned_tests = 0;
    iter_changes = 0;
    delta_BB = zeros(d, c);
    dBB_norms = zeros(1, c);
    cluster_dist = pairwiseClusterDistances(BB, aa, c);

    for i = 1:n
        m = label(i);
        n_m = aa(m);

        if n_m <= 1
            continue;
        end

        Xi = X(:, i);
        XX_i = XX(i);
        r_i = xi_norms(i);
        xBm = Xi' * BB(:, m);
        V1_m = FXXF(m) - 2 * xBm + XX_i;
        dist_count = dist_count + 1;

        delta_m = V1_m * inv_n_table(n_m) - FXXF(m) * inv_n_table(n_m + 1);
        ub_xB(m, i) = xBm;
        best_k = m;
        best_delta = delta_m;

        for k = 1:c
            if k == m
                continue;
            end

            total_tests = total_tests + 1;
            n_k = aa(k);
            FXXF_k = FXXF(k);

            if n_k == 0
                delta_k = -XX_i;
                if delta_k < best_delta
                    best_delta = delta_k;
                    best_k = k;
                end
                continue;
            end

            d_mk_safe = cluster_dist(m, k) + dBB_norms(m) + dBB_norms(k);
            ub_tight = xBm + r_i * d_mk_safe;
            ub_xB(k, i) = ub_tight;
            ub_eff = ub_tight + r_i * dBB_norms(k);
            V2_max = FXXF_k + 2 * ub_eff + XX_i;
            delta_k_min = FXXF_k * inv_n_table(n_k + 1) - V2_max * inv_n_table(n_k + 2);

            if delta_k_min > best_delta
                pruned_tests = pruned_tests + 1;
                continue;
            end

            xBk = Xi' * BB(:, k);
            dist_count = dist_count + 1;
            V2_k = FXXF_k + 2 * xBk + XX_i;
            delta_k = FXXF_k * inv_n_table(n_k + 1) - V2_k * inv_n_table(n_k + 2);
            ub_xB(k, i) = xBk;

            if delta_k < best_delta
                best_delta = delta_k;
                best_k = k;
            end
        end

        if best_k ~= m
            q = best_k;
            BB(:, q) = BB(:, q) + Xi;
            BB(:, m) = BB(:, m) - Xi;
            delta_BB(:, q) = delta_BB(:, q) + Xi;
            delta_BB(:, m) = delta_BB(:, m) - Xi;
            dBB_norms(q) = norm(delta_BB(:, q));
            dBB_norms(m) = norm(delta_BB(:, m));
            aa(q) = aa(q) + 1;
            aa(m) = aa(m) - 1;
            FXXF(m) = V1_m;
            FXXF(q) = BB(:, q)' * BB(:, q);
            ub_xB(q, i) = Xi' * BB(:, q);
            if aa(m) > 0
                ub_xB(m, i) = Xi' * BB(:, m);
            end
            label(i) = q;
            change_count = change_count + 1;
            iter_changes = iter_changes + 1;
        end
    end

    active_k_idx = find(dBB_norms > 1e-12);
    for k_idx = 1:numel(active_k_idx)
        k = active_k_idx(k_idx);
        ub_xB(k, :) = ub_xB(k, :) + dBB_norms(k) * xi_norms;
    end

    obj(iter + 1) = currentObjective(total_xx, FXXF, aa);
    iter_num = iter;

    if total_tests == 0
        update_record(iter) = 0;
    else
        update_record(iter) = pruned_tests / total_tests;
    end

    if abs(obj(iter + 1) - obj(iter)) < 1e-5 || iter_changes == 0
        obj(iter + 2:end) = obj(iter + 1);
        break;
    end
end

elapsed_time = toc(run_time);
minO = min(obj);
Y = label;

fprintf('Total changes: %d\n', change_count);
fprintf('Elapsed time: %.4f seconds\n', elapsed_time);
end

function [BB, aa, FXXF] = buildStats(X, label, c)
[d, ~] = size(X);
BB = zeros(d, c);
aa = zeros(1, c);

for k = 1:c
    idx = label == k;
    aa(k) = sum(idx);
    if aa(k) > 0
        BB(:, k) = sum(X(:, idx), 2);
    end
end

FXXF = sum(BB.^2, 1);
end

function cluster_dist = pairwiseClusterDistances(BB, aa, c)
cluster_dist = zeros(c, c);
active_idx = find(aa > 0);

for ii = 1:numel(active_idx)
    m = active_idx(ii);
    for jj = ii + 1:numel(active_idx)
        k = active_idx(jj);
        d_val = norm(BB(:, m) - BB(:, k));
        cluster_dist(m, k) = d_val;
        cluster_dist(k, m) = d_val;
    end
end
end

function value = currentObjective(total_xx, FXXF, aa)
valid = aa > 0;
value = total_xx - sum(FXXF(valid) ./ aa(valid));
end
