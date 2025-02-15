import os
import pickle as pkl

import numpy as np


def spat_approx_1a(deltax, solutions):
    return (solutions[2] - 2 * solutions[1] + solutions[0]) / np.power(deltax, 2)


# def time_approx_1a(deltat, time, func, x):
#     return (func(x, time+deltat) - 2*func(x, time) + func(x, time-deltat))/np.power(deltat, 2)


def initialize_wave(which_one, L, N):
    def b_one(x):
        return np.sin(2 * np.pi * x)

    def b_two(x):
        return np.sin(5 * np.pi * x)

    def b_three(x):
        if x > 1 / 5 and x < 2 / 5:
            return np.sin(5 * np.pi * x)
        else:
            return 0

    deltax = L / N
    xs = np.arange(0, L, deltax)

    if which_one == 1:
        func = b_one
    elif which_one == 2:
        func = b_two
    elif which_one == 3:
        func = b_three
    else:
        raise ValueError(
            f"invalid option {which_one}, choose option 1, 2 or 3 (integer form)"
        )

    # saving the solutions
    sols_prev = [func(xje) for xje in xs]
    sols = sols_prev.copy()

    sols_next = np.zeros(len(xs))

    return (sols_prev, sols, sols_next), xs, deltax


def wave_step_function(all_sols, c, xs, deltax, deltat):
    # the full function
    sols_prev, sols, sols_next = all_sols
    for j, x in enumerate(xs):
        if j == 0:
            sols_next[j] = 0
        elif j == len(xs) - 1:
            sols_next[j] = 0
        else:
            sols_next[j] = (
                np.power(deltat, 2)
                * np.power(c, 2)
                * spat_approx_1a(deltax, (sols[j - 1], sols[j], sols[j + 1]))
                + 2 * sols[j]
                - sols_prev[j]
            )
    sols_prev = sols_next.copy()
    sols = sols_next.copy()
    return sols_prev, sols, sols_next


def one_b_wrapper(which_one):
    L = 1
    N = 100
    c = 1
    deltat = 0.001
    overall_solutions = []
    soltjes, xs, deltax = initialize_wave(which_one, L, N)
    overall_solutions.append(soltjes[1])
    for i in range(30000):
        soltjes = wave_step_function(soltjes, c, xs, deltax, deltat)
        # soltjes = soltjes_new[0].copy(), soltjes_new[1].copy(), soltjes_new[2].copy()
        if i % 3000 == 0:
            overall_solutions.append(soltjes[1])
    return overall_solutions, xs


def two_dimensional_step_wave_function(all_sols, c, xs, deltax, deltat):
    pass


def initialize_grid(N):
    grid = np.zeros((N, N))

    grid[0, :] = 0  # bottom boundary
    grid[N - 1, :] = 1  # top boundary

    return grid


def apply_periodic_boundary(grid):
    grid[:, 0] = grid[:, -2]
    grid[:, -1] = grid[:, 1]


def update(grid, num_steps, N, gamma, dt, comparison=False):
    all_grids = [grid.copy()]
    print(f"this is the curent working directory: {os.getcwd()}")
    t = 0
    times = [0]

    time_appended = set()  # Define this outside the loop

    key_times = {0.001, 0.01, 0.1, 1.0}

    for n in range(num_steps):
        c_new = grid.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                c_new[i, j] = grid[i, j] + gamma * (
                    grid[i + 1, j]
                    + grid[i - 1, j]
                    + grid[i, j + 1]
                    + grid[i, j - 1]
                    - 4 * grid[i, j]
                )

        apply_periodic_boundary(c_new)

        grid[:] = c_new[:]

        t = round(t + dt, 6)
        print(f"Step: {n}, Time: {t}")

        if comparison:
            for key_t in key_times:
                if np.isclose(t, key_t, atol=1e-9) and t not in time_appended:
                    all_grids.append(c_new.copy())
                    times.append(t)
                    time_appended.add(t)
        else:
            if n % 100 == 0:
                all_grids.append(grid.copy())
                times.append(t)

    if comparison:
        path = "data/2D_diffusion_comparison.pkl"
    else:
        path = "data/2D_diffusion.pkl"

    pkl.dump(
        (all_grids, times),
        open(path, "wb"),
    )

    return all_grids, times


def run_simulation_without_animation():
    N = 100
    c = initialize_grid(N)
    L = 1.0
    D = 1
    all_grids = [c]

    dx = L / N
    dt = 0.25 * dx**2

    T_total = 1.0
    num_steps = T_total / dt

    t = 0
    times = [0]

    gamma = (D * dt) / (dx**2)

    T_total = 1.0
    num_steps = int(T_total / dt)

    if os.path.exists("Scientific_Computing_1/data/2D_diffusion.pkl"):
        all_grids, times = pkl.load(
            open("Scientific_Computing_1/data/2D_diffusion.pkl", "rb")
        )
    else:
        all_grids, times = update(c, num_steps, N, gamma, dt)

    return all_grids, times
