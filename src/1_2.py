import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def initialize_grid(N):
    grid = np.zeros((N, N))
    grid[0, :] = 1
    grid[-1, :] = 0

    return grid


def apply_periodic_boundary(grid):
    grid[:, 0] = grid[:, -2]
    grid[:, -1] = grid[:, 1]


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

        all_grids.append(c_new)

        c = c_new

        t += dt

    fig, ax = plt.subplots(figsize=(7, 7))

    c_plot = ax.pcolormesh(c, cmap="viridis", edgecolors="k", linewidth=0.5)

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
    interval = 50

    T_total = 1.0
    num_steps = int(T_total / dt)

    def update(grid):
        all_grids = [grid.copy()]

        t = 0
        times = [0]
        for n in range(num_steps):
            c_new = grid.copy()
            for i in range(1, N - 1):
                for j in range(1, N - 1):
                    c_new[i, j] = grid[i, j] + D * dt * (
                        (grid[i + 1, j] - 2 * grid[i, j] + grid[i - 1, j]) / dx**2
                        + (grid[i, j + 1] - 2 * grid[i, j] + grid[i, j - 1]) / dx**2
                    )
            # c_new[0] = 0  # Bottom boundary
            # c_new[-1] = 1  # Top boundary

            c_new[-1, :] = 0
            c_new[0, :] = 1
            apply_periodic_boundary(c_new)

            grid[:] = c_new[:]

            t = round(t + dt, 6)

            # if n % 100 == 0:
            #     all_grids.append(grid.copy())

            if (
                np.isclose(t, 0.001, atol=dt)
                or np.isclose(t, 0.01, atol=dt)
                or np.isclose(t, 0.1, atol=dt)
                or np.isclose(t, 1.0, atol=dt)
            ):
                all_grids.append(c_new.copy())
                times.append(t)

        return all_grids, times
        # pkl.dump(
        #     all_grids,
        #     open(f"Scientific_Computing_1/data/2D_diffusion_{num_steps}.pkl", "wb"),
        # )

    def animate_2f(update_func, grid, num_steps, interval=50):
        path = f"Scientific_Computing_1/data/2D_diffusion_{num_steps}.pkl"
        if os.path.exists(path):
            grids = pkl.load(open(path, "rb"))
        else:
            grids = update_func(grid)

        fig, axs = plt.subplots(figsize=(6, 6))
        img = axs.imshow(
            grids[0], cmap="viridis", interpolation="nearest", origin="lower"
        )
        plt.colorbar(img, ax=axs, label="Concentration")
        axs.set_title("2D Diffusion Simulation")

        def animate(frame):
            img.set_array(grids[frame])
            axs.set_title(f"2D Diffusion Simulation (Step: {frame * 100})")

        animation = FuncAnimation(
            fig, animate, frames=len(grids), interval=interval, blit=False
        )
        # animation.save("plots/2D_diffusion.gif", fps=50, writer="ffmpeg")
        plt.show()

    def analytical_solution(y, t):
        # Exact solution of the diffusion equation with these boundary conditions
        series_sum = np.zeros_like(y)
        for n in range(1, 100, 2):  # Sum over odd terms
            term = (
                (4 / (np.pi * n))
                * np.sin(n * np.pi * y)
                * np.exp(-D * (n * np.pi) ** 2 * t)
            )
            series_sum += term
        return y - series_sum  # Analytical solution

    c = initialize_grid(N)
    # animate_2f(update, c, num_steps, interval)
    all_c, times = update(c)

    y_values = np.linspace(0, L, N)

    # Plot simulation results vs. analytical solution
    plt.figure(figsize=(8, 6))

    for c_profile, t in zip(all_c, times):
        if t in [0.001, 0.01, 0.1, 1.0]:  # Only label key time steps
            plt.plot(y_values, c_profile, label=f"Simulation t={t:.3f}")

            plt.plot(
                y_values,
                analytical_solution(y_values, t),
                "--",
                label=f"Analytical t={t:.3f}",
            )
    plt.xlabel("y (Position)")
    plt.ylabel("Concentration c(y, t)")
    plt.legend()
    plt.title("Diffusion Simulation vs Analytical Solution")
    plt.show()


# run_simulation_without_animation()

run_simulation_with_animation()
