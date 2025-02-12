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
    grid[0, :] = 1
    grid[-1, :] = 0

    return grid


def apply_periodic_boundary(grid):
    grid[:, 0] = grid[:, -2]
    grid[:, -1] = grid[:, 1]


N = 1
c = initialize_grid(N)
L = 1.0
D = 1
num_steps = 5000


dx = L / N
dt = 0.25 * dx**2


def update(c):
    num_steps = 500
    for n in range(num_steps):
        c_new = c.copy()

        # Update interior points
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                c_new[i, j] = c[i, j] + D * dt * (
                    (c[i + 1, j] - 2 * c[i, j] + c[i - 1, j]) / dx**2
                    + (c[i, j + 1] - 2 * c[i, j] + c[i, j - 1]) / dx**2
                )

    # Apply boundary conditions
    c_new[N - 1, :] = 1
    c_new[0, :] = 0
    apply_periodic_boundary(c_new)  # Make sure of the periodic boundary conditions

    return c_new


# fig, ax = plt.subplots(figsize=(7, 7))

# c_plot = ax.pcolormesh(c, cmap="viridis", edgecolors="k", linewidth=0.5)

# # plt.imshow(c, cmap="viridis", interpolation="nearest", origin="lower")
# plt.colorbar(c_plot, ax=ax, label="Concentration")
# ax.set_xticks(np.arange(N))
# ax.set_yticks(np.arange(N))
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title("2D Diffusion")
# plt.show()
