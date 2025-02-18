import os
import pickle as pkl
from math import erfc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def initialize_grid(N):
    grid = np.zeros((N, N))
    # grid[0, :] = 1
    # grid[-1, :] = 0

    grid[0, :] = 0  # bottom boundary
    grid[N - 1, :] = 1  # top boundary

    return grid


def apply_periodic_boundary(grid):
    grid[:, 0] = grid[:, -2]
    grid[:, -1] = grid[:, 1]


def update(grid, num_steps, N, gamma, dt):
    all_grids = [grid.copy()]

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
        # print(f"Step: {n}, Time: {t}")

        # if n % 100 == 0:
        #     all_grids.append(grid.copy())

        for key_t in key_times:
            if np.isclose(t, key_t, atol=1e-6) and t not in time_appended:
                all_grids.append(c_new.copy())
                times.append(t)
                time_appended.add(t)

    print(f"Time steps: {times}")
    pkl.dump(
        (all_grids, times),
        open("Scientific_Computing_1/data/2D_diffusion.pkl", "wb"),
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
        # all_grids, times = update(c, num_steps, N, gamma, dt)
        pass

    fig, ax = plt.subplots(figsize=(7, 7))

    c_plot = ax.pcolormesh(all_grids[-1], cmap="viridis", edgecolors="k", linewidth=0.5)

    # plt.imshow(c, cmap="viridis", interpolation="nearest", origin="lower")
    plt.colorbar(c_plot, ax=ax, label="Concentration")
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("2D Diffusion")
    plt.show()


def run_simulation_with_animation():
    N = 100
    c = initialize_grid(N)
    L = 1.0
    D = 1

    dx = L / N
    dt = 0.25 * dx**2

    gamma = (D * dt) / (dx**2)

    T_total = 1.0
    num_steps = int(T_total / dt)

    # print(f"starting c: {c}")
    # plt.imshow(c, cmap="viridis", interpolation="nearest", origin="lower")
    # plt.show()

    # def update(grid):
    #     all_grids = [grid.copy()]

    #     t = 0
    #     times = [0]

    #     time_appended = set()  # Define this outside the loop

    #     key_times = {0.001, 0.01, 0.1, 1.0}

    #     for n in range(num_steps):
    #         c_new = grid.copy()
    #         for i in range(1, N - 1):
    #             for j in range(1, N - 1):
    #                 c_new[i, j] = grid[i, j] + gamma * (
    #                     grid[i + 1, j]
    #                     + grid[i - 1, j]
    #                     + grid[i, j + 1]
    #                     + grid[i, j - 1]
    #                     - 4 * grid[i, j]
    #                 )

    #         apply_periodic_boundary(c_new)

    #         grid[:] = c_new[:]

    #         t = round(t + dt, 6)
    #         print(f"Step: {n}, Time: {t}")

    #         # if n % 100 == 0:
    #         #     all_grids.append(grid.copy())

    #         for key_t in key_times:
    #             if np.isclose(t, key_t, atol=1e-6) and t not in time_appended:
    #                 all_grids.append(c_new.copy())
    #                 times.append(t)
    #                 time_appended.add(t)

    #     # print(f"Time steps: {times}")
    #     pkl.dump(
    #         (all_grids, times),
    #         open("Scientific_Computing_1/data/2D_diffusion.pkl", "wb"),
    #     )

    #     return all_grids, times

    def animate_2f(update_func, grid, num_steps, N, gamma, dt, interval=50):
        print(os.getcwd())
        path = "Scientific_Computing_1/data/2D_diffusion_comparison.pkl"
        if os.path.exists(path):
            grids, times = pkl.load(open(path, "rb"))
        else:
            grids, times = update_func(grid, num_steps, N, gamma, dt)

        print(f"Starting grid: {grid}")

        fig, axs = plt.subplots(figsize=(6, 6))
        img = axs.imshow(
            grids[0], cmap="viridis", interpolation="nearest", origin="lower"
        )
        plt.colorbar(img, ax=axs, label="Concentration")
        axs.set_title("2D Diffusion Simulation")

        def animate(frame):
            img.set_array(grids[frame])
            axs.set_title(f"2D Diffusion Simulation (Step: {frame:.3g})")

        animation = FuncAnimation(
            fig, animate, frames=len(grids), interval=interval, blit=False
        )
        # animation.save("plots/2D_diffusion.gif", fps=50, writer="ffmpeg")
        plt.show()

    def analytical_solution(x, t, D=1, i_max=100):
        if t <= 0:
            return np.nan

        sum_val = 0.0
        for i in range(i_max + 1):
            arg_1 = (1 - x + 2 * i) / (2 * np.sqrt(D * t))
            arg_2 = (1 + x + 2 * i) / (2 * np.sqrt(D * t))
            sum_val += erfc(arg_1) - erfc(arg_2)

        return sum_val

    def plot_with_error(y, solutions, times, D):
        """
        Funcion comparing analytical solution to numerical solution.
        Plotting points for numerical solution and line for analytical soltution
        """
        nx = 100
        y_analytic = np.linspace(0.0, 1.0, nx)

        # plt.figure(figsize=(4.5, 3))

        fig1, ax1 = plt.subplots(figsize=(4.5, 3))
        fig2, ax2 = plt.subplots(figsize=(4.5, 3))
        colors = ["orange", "blue", "green", "purple", "brown"]

        for i, t_val in enumerate(times):
            # Analytical
            c_analytical = [analytical_solution(yy, t_val, D=D) for yy in y_analytic]
            ax1.plot(
                y_analytic,
                c_analytical,
                color=colors[i],
                linewidth=0.8,
                label=f"t={t_val:.3g}",
            )

            c_analytical = np.array(c_analytical)

            # mean of the row (should theoretically all be the same)
            c2D = solutions[i]
            c_first = c2D[:, 0]
            c_first = np.array(c_first)

            # Absolute error
            pointwise_diff = np.abs(c_first - c_analytical)

            # slice array to avoid clutteredness
            ax1.plot(y[::2], c_first[::2], "o", color=colors[i], markersize=2)

            ax2.plot(
                y[::2],
                pointwise_diff[::2],
                "o",
                label=f"t={t_val}",
                color=colors[i],
                markersize=2,
            )

        ax1.set_xlabel("y (Position)")
        ax1.set_ylabel("Concentration c(y, t)")
        ax1.legend()
        ax1.set_title("Diffusion Simulation vs Analytical Solution")
        ax1.grid(True)
        fig1.tight_layout()
        # fig1.savefig("plots/diffusion_analytical.png", dpi=300, bbox_inches="tight")

        ax2.set_xlabel("y (Position)")
        ax2.set_ylabel("Absolute error")
        ax2.legend()
        ax2.set_title("Error Simulation and Analytical Solution")
        ax2.grid(True)
        fig2.tight_layout()
        # fig2.savefig(
        #     "plots/diffusion_analytical_error.png", dpi=300, bbox_inches="tight"
        # )

        plt.show()

    def plot_analytical_solution(y, solutions, times, D):
        nx = 100
        y_analytic = np.linspace(0.0, 1.0, nx)

        plt.figure(figsize=(7, 5))

        for i, t_val in enumerate(times):
            # Analytical
            c_analytical = [analytical_solution(yy, t_val, D=D) for yy in y_analytic]
            plt.plot(y_analytic, c_analytical, label=f"Analytical t={t_val:.3g}")

            c_analytical = np.array(c_analytical)
            c2D = solutions[i]
            c_avg = np.mean(c2D, axis=1)

            plt.plot(y, c_avg, "o", label=f"Simulation t={t_val:.3g}")

        plt.xlabel("y (Position)")
        plt.ylabel("Concentration c(y, t)")
        plt.legend()
        plt.title("Diffusion Simulation vs Analytical Solution")
        plt.grid(True)
        plt.show()

    # Main part of the function

    # c = initialize_grid(N)
    print(c)
    # animate_2f(update, c, num_steps, interval)
    print(os.getcwd())
    if os.path.exists("Scientific_Computing_1/data/2D_diffusion.pkl"):
        all_c, times = pkl.load(
            open("Scientific_Computing_1/data/2D_diffusion.pkl", "rb")
        )
    else:
        all_c, times = update(c)

    y_values = np.linspace(0, 1, N)

    # Plot simulation results vs analytical solution
    plot_with_error(y_values, all_c, times, D)
    # animate_2f(update, c, num_steps, N, gamma, dt)


#   run_simulation_without_animation()


def plot_five_states():
    N = 100
    c = initialize_grid(N)

    L = 1.0
    D = 1

    dx = L / N
    dt = 0.25 * dx**2

    gamma = (D * dt) / (dx**2)

    T_total = 1.0
    num_steps = int(T_total / dt)

    if os.path.exists("Scientific_Computing_1/data/2D_diffusion.pkl"):
        all_c, times = pkl.load(
            open("Scientific_Computing_1/data/2D_diffusion.pkl", "rb")
        )
    else:
        all_c, times = update(c, num_steps, N, gamma, dt)

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("2D Diffusion Simulation at t = 0, 0.001, 0.01, 0.1, 1.0")

    print(all_c[3])

    axs[0].imshow(all_c[0], cmap="viridis", interpolation="nearest", origin="lower")
    axs[0].set_title(f"t = {times[0]}")

    axs[1].imshow(all_c[1], cmap="viridis", interpolation="nearest", origin="lower")
    axs[1].set_title(f"t = {times[1]}")

    axs[2].imshow(all_c[2], cmap="viridis", interpolation="nearest", origin="lower")
    axs[2].set_title(f"t = {times[2]}")

    axs[3].imshow(all_c[3], cmap="viridis", interpolation="nearest", origin="lower")
    axs[3].set_title(f"t = {times[3]}")

    axs[4].imshow(all_c[4], cmap="viridis", interpolation="nearest", origin="lower")
    axs[4].set_title(f"t = {times[4]}")

    plt.show()


# plot_five_states()
run_simulation_with_animation()
