import numpy as np
import sympy as sp


def conv_mutation(points, vals):
    weights = np.random.rand(3)
    weights = np.sort(weights)
    weights = weights[np.argsort(-vals)] / weights.sum()
    return (weights * points).sum()


def check_conv(interval, a, b, c):
    if interval == sp.EmptySet:
        return False
    if isinstance(interval, sp.Interval):
        return interval.contains(a) and interval.contains(b) and interval.contains(c)
    for arg in interval.args:
        if arg.contains(a) and arg.contains(b) and arg.contains(c):
            return True
    return False


def conv_differential_evolution_1d(f, bounds, pop_size=5, max_generations=1000, F=0.5, CR=0.4, ConvCR=0.25, tol=1e-6, true_mn=0):
    x = sp.Symbol('x', real=True)
    f_eval = sp.lambdify(x, f.func)
    pop = np.random.uniform(bounds[0], bounds[1], size=pop_size)
    for i, point in enumerate(pop):
        if point in f.concavity_intervals:
            pop[i] = np.random.choice(list(f.concavity_intervals.boundary), size=1)[0]

    fitness = np.array([f_eval(e) for e in pop])

    best_idx = np.argmin(fitness)
    best_sol = pop[best_idx]
    best_val = fitness[best_idx]

    for i in range(max_generations):

        trial_pop = np.zeros((pop_size,))
        for j in range(pop_size):

            a, b, c = np.random.choice(pop_size, size=3, replace=False)

            if check_conv(f.convexity_intervals, pop[a], pop[b], pop[c]) and np.random.rand(1) < ConvCR:
                trial_pop[j] = conv_mutation(pop[[a, b, c]], fitness[[a, b, c]])
            else:
                trial_pop[j] = pop[a] + F * (pop[b] - pop[c])

            trial_pop[j] = np.clip(trial_pop[j], bounds[0], bounds[1])
        for j, point in enumerate(trial_pop):
            if point in f.concavity_intervals:
                trial_pop[j] = np.random.choice(list(f.concavity_intervals.boundary), size=1)[0]
        trial_fitness = np.array([f_eval(x) for x in trial_pop])

        crossover_mask = np.random.rand(pop_size) < CR
        trial_pop = np.where(crossover_mask, trial_pop, pop)
        trial_fitness = np.where(crossover_mask, trial_fitness, fitness)

        selection_mask = trial_fitness < fitness
        pop[selection_mask] = trial_pop[selection_mask]
        fitness[selection_mask] = trial_fitness[selection_mask]

        new_best_idx = np.argmin(fitness)
        new_best_val = fitness[new_best_idx]
        if new_best_val < best_val:
            best_idx = new_best_idx
            best_sol = pop[best_idx]
            best_val = new_best_val

        if np.abs(best_val - true_mn) < 1e-5:
            return best_sol, best_val, i + 1

        if np.std(pop) < tol:
            break
    return best_sol, best_val, np.NaN


