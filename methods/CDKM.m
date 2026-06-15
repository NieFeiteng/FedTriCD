function [Y, minO, iter_num, obj, elapsed_time, update_record, dist_count] = CDKM(X, label, c, iter_rounds)
fprintf('CDKM\n');

label = label(:);
[~, n] = size(X);

XX = sum(X.^2, 1);
total_xx = sum(XX);
[BB, aa, FXXF] = buildStats(X, label, c);

obj = zeros(1, iter_rounds + 1);
obj(1) = currentObjective(total_xx, FXXF, aa);
update_record = zeros(1, iter_rounds);
iter_num = 0;
dist_count = 0;
change_count = 0;

run_time = tic;

for iter = 1:iter_rounds
    iter_updates = 0;

    for i = 1:n
        m = label(i);

        if aa(m) <= 1
            continue;
        end

        Xi = X(:, i);
        XX_i = XX(i);
        delta = inf(1, c);
        V1_m = 0;
        V2 = zeros(1, c);

        for k = 1:c
            xBk = Xi' * BB(:, k);

            if k == m
                V1_m = FXXF(k) - 2 * xBk + XX_i;
                delta(k) = V1_m / (aa(k) - 1) - FXXF(k) / aa(k);
            else
                V2(k) = FXXF(k) + 2 * xBk + XX_i;
                old_term = 0;
                if aa(k) > 0
                    old_term = FXXF(k) / aa(k);
                end
                delta(k) = old_term - V2(k) / (aa(k) + 1);
            end

            dist_count = dist_count + 1;
        end

        [~, q] = min(delta);

        if q ~= m
            BB(:, q) = BB(:, q) + Xi;
            BB(:, m) = BB(:, m) - Xi;
            aa(q) = aa(q) + 1;
            aa(m) = aa(m) - 1;
            FXXF(m) = V1_m;
            FXXF(q) = V2(q);
            label(i) = q;
            change_count = change_count + 1;
            iter_updates = iter_updates + 1;
        end
    end

    update_record(iter) = iter_updates;
    obj(iter + 1) = currentObjective(total_xx, FXXF, aa);
    iter_num = iter;

    if abs(obj(iter + 1) - obj(iter)) < 1e-5
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

function value = currentObjective(total_xx, FXXF, aa)
valid = aa > 0;
value = total_xx - sum(FXXF(valid) ./ aa(valid));
end
