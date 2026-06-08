import math

import numpy as np
from numba import njit


@njit(cache=True)
def predict_value(aid, bid, cid, U, V, W, D, E, F, rank):
    pred = 0.0

    for r1 in range(1, rank + 1):
        for r2 in range(1, rank + 1):
            for r3 in range(1, rank + 1):
                pred += U[r3, aid, r1] * V[r1, bid, r2] * W[r2, cid, r3]

    for r in range(1, rank + 1):
        pred += D[aid, r] * E[bid, r] * F[cid, r]

    return pred


@njit(cache=True)
def evaluate(indices, values, U, V, W, D, E, F, rank):
    square = 0.0
    abs_sum = 0.0
    n = values.shape[0]

    for idx in range(n):
        aid = indices[idx, 0]
        bid = indices[idx, 1]
        cid = indices[idx, 2]
        pred = predict_value(aid, bid, cid, U, V, W, D, E, F, rank)
        diff = values[idx] - pred
        square += diff * diff
        abs_sum += diff if diff >= 0.0 else -diff

    return math.sqrt(square / n), abs_sum / n


@njit(cache=True)
def one_particle_update(
    train_indices,
    train_values,
    U,
    V,
    W,
    D,
    E,
    F,
    Uup,
    Udown,
    Vup,
    Vdown,
    Wup,
    Wdown,
    Dup,
    Ddown,
    Eup,
    Edown,
    Fup,
    Fdown,
    rank,
    max_aid,
    max_bid,
    max_cid,
    lambda_reg,
    lambda_b,
):
    Uup.fill(0.0)
    Udown.fill(0.0)
    Vup.fill(0.0)
    Vdown.fill(0.0)
    Wup.fill(0.0)
    Wdown.fill(0.0)
    Dup.fill(0.0)
    Ddown.fill(0.0)
    Eup.fill(0.0)
    Edown.fill(0.0)
    Fup.fill(0.0)
    Fdown.fill(0.0)

    n_train = train_values.shape[0]

    for idx in range(n_train):
        aid = train_indices[idx, 0]
        bid = train_indices[idx, 1]
        cid = train_indices[idx, 2]
        x = train_values[idx]
        x_hat = predict_value(aid, bid, cid, U, V, W, D, E, F, rank)

        for r1 in range(1, rank + 1):
            for r3 in range(1, rank + 1):
                temp = 0.0
                for r2 in range(1, rank + 1):
                    temp += V[r1, bid, r2] * W[r2, cid, r3]
                Uup[r3, aid, r1] += x * temp
                Udown[r3, aid, r1] += x_hat * temp + lambda_reg * U[r3, aid, r1]

    for aid in range(1, max_aid + 1):
        for r1 in range(1, rank + 1):
            for r3 in range(1, rank + 1):
                if Udown[r3, aid, r1] != 0.0:
                    U[r3, aid, r1] = U[r3, aid, r1] * Uup[r3, aid, r1] / Udown[r3, aid, r1]

    for idx in range(n_train):
        aid = train_indices[idx, 0]
        bid = train_indices[idx, 1]
        cid = train_indices[idx, 2]
        x = train_values[idx]
        x_hat = predict_value(aid, bid, cid, U, V, W, D, E, F, rank)

        for r1 in range(1, rank + 1):
            for r2 in range(1, rank + 1):
                temp = 0.0
                for r3 in range(1, rank + 1):
                    temp += U[r3, aid, r1] * W[r2, cid, r3]
                Vup[r1, bid, r2] += x * temp
                Vdown[r1, bid, r2] += x_hat * temp + lambda_reg * V[r1, bid, r2]

    for bid in range(1, max_bid + 1):
        for r1 in range(1, rank + 1):
            for r2 in range(1, rank + 1):
                if Vdown[r1, bid, r2] != 0.0:
                    V[r1, bid, r2] = V[r1, bid, r2] * Vup[r1, bid, r2] / Vdown[r1, bid, r2]

    for idx in range(n_train):
        aid = train_indices[idx, 0]
        bid = train_indices[idx, 1]
        cid = train_indices[idx, 2]
        x = train_values[idx]
        x_hat = predict_value(aid, bid, cid, U, V, W, D, E, F, rank)

        for r3 in range(1, rank + 1):
            for r2 in range(1, rank + 1):
                temp = 0.0
                for r1 in range(1, rank + 1):
                    temp += U[r3, aid, r1] * V[r1, bid, r2]
                Wup[r2, cid, r3] += x * temp
                Wdown[r2, cid, r3] += x_hat * temp + lambda_reg * W[r2, cid, r3]

    for cid in range(1, max_cid + 1):
        for r2 in range(1, rank + 1):
            for r3 in range(1, rank + 1):
                if Wdown[r2, cid, r3] != 0.0:
                    W[r2, cid, r3] = W[r2, cid, r3] * Wup[r2, cid, r3] / Wdown[r2, cid, r3]

    for idx in range(n_train):
        aid = train_indices[idx, 0]
        bid = train_indices[idx, 1]
        cid = train_indices[idx, 2]
        x = train_values[idx]
        x_hat = predict_value(aid, bid, cid, U, V, W, D, E, F, rank)

        for r in range(1, rank + 1):
            temp = E[bid, r] * F[cid, r]
            Dup[aid, r] += x * temp
            Ddown[aid, r] += x_hat * temp + lambda_b * D[aid, r]

    for aid in range(1, max_aid + 1):
        for r in range(1, rank + 1):
            if Ddown[aid, r] != 0.0:
                D[aid, r] = D[aid, r] * Dup[aid, r] / Ddown[aid, r]

    for idx in range(n_train):
        aid = train_indices[idx, 0]
        bid = train_indices[idx, 1]
        cid = train_indices[idx, 2]
        x = train_values[idx]
        x_hat = predict_value(aid, bid, cid, U, V, W, D, E, F, rank)

        for r in range(1, rank + 1):
            temp = D[aid, r] * F[cid, r]
            Eup[bid, r] += x * temp
            Edown[bid, r] += x_hat * temp + lambda_b * E[bid, r]

    for bid in range(1, max_bid + 1):
        for r in range(1, rank + 1):
            if Edown[bid, r] != 0.0:
                E[bid, r] = E[bid, r] * Eup[bid, r] / Edown[bid, r]

    for idx in range(n_train):
        aid = train_indices[idx, 0]
        bid = train_indices[idx, 1]
        cid = train_indices[idx, 2]
        x = train_values[idx]
        x_hat = predict_value(aid, bid, cid, U, V, W, D, E, F, rank)

        for r in range(1, rank + 1):
            temp = D[aid, r] * E[bid, r]
            Fup[cid, r] += x * temp
            Fdown[cid, r] += x_hat * temp + lambda_b * F[cid, r]

    for cid in range(1, max_cid + 1):
        for r in range(1, rank + 1):
            if Fdown[cid, r] != 0.0:
                F[cid, r] = F[cid, r] * Fup[cid, r] / Fdown[cid, r]


@njit(cache=True)
def train_kernel(
    train_indices,
    train_values,
    valid_indices,
    valid_values,
    test_indices,
    test_values,
    U,
    V,
    W,
    D,
    E,
    F,
    rank,
    max_aid,
    max_bid,
    max_cid,
    train_round,
    threshold,
    errorgap,
    print_every,
    px,
    pv,
    min_x,
    max_x,
    min_v,
    max_v,
    rand1,
    rand2,
    c1,
    c2,
    w,
    alpha,
    use_current_validation_fitness,
):
    every_round_rmse = np.zeros(train_round + 1, dtype=np.float64)
    every_round_mae = np.zeros(train_round + 1, dtype=np.float64)
    every_round_rmse_test = np.zeros(train_round + 1, dtype=np.float64)
    every_round_mae_test = np.zeros(train_round + 1, dtype=np.float64)
    best_lambda_history = np.zeros((train_round + 1, 2), dtype=np.float64)

    every_round_rmse[0] = 100.0
    every_round_mae[0] = 100.0
    every_round_rmse_test[0] = 100.0
    every_round_mae_test[0] = 100.0

    init_rmse, init_mae = evaluate(valid_indices, valid_values, U, V, W, D, E, F, rank)
    init_fitness = alpha * init_rmse + (1.0 - alpha) * init_mae

    population = px.shape[0]
    p_best = np.empty(population, dtype=np.float64)
    p_best_value = np.zeros((population, 2), dtype=np.float64)
    for p in range(population):
        p_best[p] = -1.0e100
        for h in range(2):
            p_best_value[p, h] = px[p, h]

    g_best = -1.0e100
    g_best_value = np.zeros(2, dtype=np.float64)
    for h in range(2):
        g_best_value[h] = px[0, h]

    min_rmse = 100.0
    min_mae = 100.0
    min_rmse_round = 0
    min_mae_round = 0
    stable_count = 0
    stop_round = train_round

    Uup = np.zeros_like(U)
    Udown = np.zeros_like(U)
    Vup = np.zeros_like(V)
    Vdown = np.zeros_like(V)
    Wup = np.zeros_like(W)
    Wdown = np.zeros_like(W)
    Dup = np.zeros_like(D)
    Ddown = np.zeros_like(D)
    Eup = np.zeros_like(E)
    Edown = np.zeros_like(E)
    Fup = np.zeros_like(F)
    Fdown = np.zeros_like(F)

    fitness = np.zeros(population, dtype=np.float64)
    fitness_norm = np.zeros(population, dtype=np.float64)

    for round_id in range(1, train_round + 1):
        flag_rmse = True
        flag_mae = True

        for p in range(population):
            lambda_reg = px[p, 0]
            lambda_b = px[p, 1]
            one_particle_update(
                train_indices,
                train_values,
                U,
                V,
                W,
                D,
                E,
                F,
                Uup,
                Udown,
                Vup,
                Vdown,
                Wup,
                Wdown,
                Dup,
                Ddown,
                Eup,
                Edown,
                Fup,
                Fdown,
                rank,
                max_aid,
                max_bid,
                max_cid,
                lambda_reg,
                lambda_b,
            )

            cur_rmse, cur_mae = evaluate(valid_indices, valid_values, U, V, W, D, E, F, rank)
            if use_current_validation_fitness:
                fitness[p] = alpha * cur_rmse + (1.0 - alpha) * cur_mae
            else:
                fitness[p] = init_fitness

        f_down = fitness[population - 1] - init_fitness
        if f_down < 0.0:
            abs_f_down = -f_down
        else:
            abs_f_down = f_down

        if abs_f_down < 1.0e-12:
            for p in range(population):
                fitness_norm[p] = -fitness[p]
        else:
            for p in range(population):
                if p == 0:
                    fitness_norm[p] = (fitness[p] - init_fitness) / f_down
                else:
                    fitness_norm[p] = (fitness[p] - fitness[p - 1]) / f_down

        for p in range(population):
            if fitness_norm[p] > p_best[p]:
                p_best[p] = fitness_norm[p]
                for h in range(2):
                    p_best_value[p, h] = px[p, h]

            if fitness_norm[p] > g_best:
                g_best = fitness_norm[p]
                for h in range(2):
                    g_best_value[h] = px[p, h]

        for p in range(population):
            r1 = rand1[round_id, p]
            r2 = rand2[round_id, p]
            for h in range(2):
                pv[p, h] = (
                    w * pv[p, h]
                    + c1 * r1 * (p_best_value[p, h] - px[p, h])
                    + c2 * r2 * (g_best_value[h] - px[p, h])
                )

                if pv[p, h] < min_v[h]:
                    pv[p, h] = min_v[h]
                if pv[p, h] > max_v[h]:
                    pv[p, h] = max_v[h]

                px[p, h] += pv[p, h]

                if px[p, h] < min_x[h]:
                    px[p, h] = min_x[h]
                if px[p, h] > max_x[h]:
                    px[p, h] = max_x[h]

        best_lambda_history[round_id, 0] = g_best_value[0]
        best_lambda_history[round_id, 1] = g_best_value[1]

        valid_rmse, valid_mae = evaluate(valid_indices, valid_values, U, V, W, D, E, F, rank)
        test_rmse, test_mae = evaluate(test_indices, test_values, U, V, W, D, E, F, rank)

        every_round_rmse[round_id] = valid_rmse
        every_round_mae[round_id] = valid_mae
        every_round_rmse_test[round_id] = test_rmse
        every_round_mae_test[round_id] = test_mae

        if print_every > 0 and round_id % print_every == 0:
            print(round_id, test_rmse, test_mae, g_best_value[0], g_best_value[1])

        if every_round_rmse[round_id - 1] - every_round_rmse[round_id] > errorgap:
            if min_rmse > every_round_rmse[round_id]:
                min_rmse = every_round_rmse[round_id]
                min_rmse_round = round_id
            flag_rmse = False
            stable_count = 0

        if every_round_mae[round_id - 1] - every_round_mae[round_id] > errorgap:
            if min_mae > every_round_mae[round_id]:
                min_mae = every_round_mae[round_id]
                min_mae_round = round_id
            flag_mae = False
            stable_count = 0

        if flag_rmse and flag_mae:
            stable_count += 1
            if threshold > 0 and stable_count == threshold:
                stop_round = round_id
                break

    return (
        every_round_rmse,
        every_round_mae,
        every_round_rmse_test,
        every_round_mae_test,
        best_lambda_history,
        min_rmse_round,
        min_mae_round,
        stop_round,
        px,
        pv,
        g_best_value,
        g_best,
    )
