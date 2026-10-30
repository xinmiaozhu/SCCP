clc,clear,close all

M = 96;
N = 256;
K = 5;
x = zeros(N,1);
idx_true = randperm(N,K);
x(idx_true) = exp(j*2*pi*rand(K,1));
A = randn(M,N) + j*randn(M,N);
y = abs(A*x);

ga_opts.pop_size   = 1e3;      
ga_opts.max_gen    = 50;   
ga_opts.p_crossover= 1;    
ga_opts.p_mutation = 0.2;     
ga_opts.tournament = 3;      
ga_opts.elite      = 2;   
ga_opts.verbose    = true;

[idx_hat, x_hat, fit_hist, best_hist] = ga_subspace_search(A, y, K, ga_opts);

fprintf('\nTrue Non-Zero Support: %s\n', mat2str(sort(idx_true)));
fprintf('Estimated Non-Zero Support: %s\n', mat2str(sort(idx_hat)'));
fprintf('Whether Rocovery is Successful: %d\n', isequal(sort(idx_true), sort(idx_hat)'));


figure;
semilogy(fit_hist, 'LineWidth', 1.2); hold on;
semilogy(best_hist, 'r', 'LineWidth', 1.5);
xlabel('Generation','interpreter','latex'); ylabel('Fitness (residual)','interpreter','latex');
legend('Average','Global Best','interpreter','latex'); grid on;
title('GA Convergence','interpreter','latex');

function [idx_best, x_best, fit_hist, best_hist] = ga_subspace_search(A, y, K, opts)

    [M, N] = size(A);

    pop = false(N, opts.pop_size);
    for i = 1:opts.pop_size
        pop(randperm(N, K), i) = true;
    end

    fit_hist  = zeros(opts.max_gen, 1);
    best_hist = zeros(opts.max_gen, 1);
    fit = zeros(opts.pop_size, 1);

    idx_best = [];
    x_best   = [];
    f_best   = inf;

    cache_f = containers.Map('KeyType','char','ValueType','double');
    cache_x = containers.Map('KeyType','char','ValueType','any');

    for gen = 1:opts.max_gen

        idx_all = cell(opts.pop_size, 1);
        key_all = cell(opts.pop_size, 1);
        for i = 1:opts.pop_size
            idx_all{i} = find(pop(:, i));
            key_all{i} = sprintf('%d,', sort(idx_all{i}));
        end

        need_compute = false(opts.pop_size, 1);
        for i = 1:opts.pop_size
            if isKey(cache_f, key_all{i})
                fit(i) = cache_f(key_all{i});
            else
                need_compute(i) = true;
            end
        end

        need_idx = find(need_compute);
        if ~isempty(need_idx)
            need_keys = key_all(need_idx);
            [uniq_keys, ~, ic] = unique(need_keys);   
            n_uniq = numel(uniq_keys);

            uniq_idx = cell(n_uniq, 1);
            for u = 1:n_uniq
                uniq_idx{u} = idx_all{need_idx(find(ic == u, 1))};
            end

            f_uniq = zeros(n_uniq, 1);
            x_uniq = cell(n_uniq, 1);
            parfor u = 1:n_uniq
                [f_uniq(u), x_uniq{u}] = fitness_subspace(A, y, uniq_idx{u});
            end

            for u = 1:n_uniq
                cache_f(uniq_keys{u}) = f_uniq(u);
                cache_x(uniq_keys{u}) = x_uniq{u};
            end
            for j = 1:numel(need_idx)
                i = need_idx(j);
                fit(i) = f_uniq(ic(j));
            end
        end

        [f_min, i_min] = min(fit);
        if f_min < f_best
            f_best   = f_min;
            idx_best = find(pop(:, i_min));
            key      = sprintf('%d,', sort(idx_best));
            x_best   = cache_x(key);
        end
        fit_hist(gen)  = mean(fit);
        best_hist(gen) = f_best;

        if opts.verbose && mod(gen, 5) == 0
            fprintf('Gen %3d | Average %.3e | Global Best %.3e | Indices = %s | Cache Length = %d\n', ...
                gen, mean(fit), f_best, mat2str(sort(idx_best)'), cache_f.Count);
        end

        [~, order] = sort(fit, 'ascend');
        new_pop = false(N, opts.pop_size);
        new_pop(:, 1:opts.elite) = pop(:, order(1:opts.elite));

        cnt = opts.elite;
        while cnt < opts.pop_size
            p1 = tournament_select(fit, opts.tournament);
            p2 = tournament_select(fit, opts.tournament);
            if rand < opts.p_crossover
                c1 = set_crossover(pop(:, p1), pop(:, p2), K);
            else
                c1 = pop(:, p1);
            end
            if rand < opts.p_mutation
                c1 = set_mutate(c1, K);
            end
            cnt = cnt + 1;
            new_pop(:, cnt) = c1;
        end

        pop = new_pop;
    end
end

function [fval, x_full] = fitness_subspace(A, y, idx)
    A_sub = A(:, idx);
    m = size(A,1);

    Y = (A_sub .* (y.^2))' * A_sub / m;
    [V, D] = eigs(Y, 1);
    x0 = sqrt(max(D,0)) * V;
    if norm(x0) == 0
        x0 = randn(numel(idx),1) + 1j*randn(numel(idx),1);
    end
    x0 = x0 / norm(x0) * sqrt(mean(y.^2));

    x_sub = gradient_descent(A_sub, y, x0);

    fval = norm(abs(A_sub * x_sub) - y) / norm(y);
    x_full = zeros(size(A,2),1);
    x_full(idx) = x_sub;
end

function p = tournament_select(fit, k)
    cand = randperm(numel(fit), k);
    [~, j] = min(fit(cand));
    p = cand(j);
end

function c = set_crossover(p1, p2, K)
    c = p1 | p2;                   
    if nnz(c) > K
        ones_idx = find(c);
        keep = ones_idx(randperm(numel(ones_idx), K));
        c = false(size(c)); c(keep) = true;
    elseif nnz(c) < K
        zeros_idx = find(~c);
        add = zeros_idx(randperm(numel(zeros_idx), K - nnz(c)));
        c(add) = true;
    end
end

function c = set_mutate(c, K)
    in_idx  = find(c);
    out_idx = find(~c);
    if isempty(out_idx) || isempty(in_idx), return; end
    drop = in_idx(randi(numel(in_idx)));
    add  = out_idx(randi(numel(out_idx)));
    c(drop) = false;
    c(add)  = true;
    assert(nnz(c) == K);
end

function x_hat = gradient_descent(A, y, x_init)
    max_iter = 10; tol = 1e-6;
    m = size(A,1);
    x_hat = x_init;

    [obj, grad] = obj_and_grad(A, y, x_hat);
    Ag = A * grad;               
    Ax = A * x_hat;

    for iter = 1:max_iter
        step = 1;
        for bt = 1:20
            Ax_c = Ax - step * Ag;
            obj_c = sum((abs(Ax_c).^2 - y.^2).^2) / (2*m);
            if obj_c < obj, break; end
            step = 0.5 * step;
        end

        x_new = x_hat - step * grad;
        if norm(x_new - x_hat) / norm(x_hat) < tol || bt >= 20
            x_hat = x_new; break;
        end
        x_hat = x_new;

        [obj, grad] = obj_and_grad(A, y, x_hat);
        Ax = A * x_hat;
        Ag = A * grad;
    end
end

function [obj, grad] = obj_and_grad(A, y, x)
    m = length(y);
    Ax = A * x;
    r  = abs(Ax).^2 - y.^2;
    obj  = sum(r.^2) / (2*m);
    grad = (2/m) * (A' * (r .* Ax));
end
