import os
import pickle as pkl
from math import erfc

import matplotlib.pyplot as plt
import numpy as np
import src.solutions as solutions
from matplotlib.animation import FuncAnimation

from .solutions import initialize_grid, update


def visualization_1b(overall_solutions, xs):
    fig, axs = plt.subplots(1, 3, figsize=(5.3, 2.5), sharey=True)

    for j in range(len(overall_solutions)):
        for k in range(len(overall_solutions[j])):
            axs[j].plot(xs, overall_solutions[j][k], linewidth=1)
        axs[j].set_xlabel("x")
        axs[j].set_title("i" * (j + 1))

    axs[0].set_ylabel(r"$\Psi^n$")

    fig.suptitle("Wave Time-stepping Approximation")
    plt.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.savefig("plots/fig_1B.png", dpi=300)
    plt.show()


def animate_1c(L, N, c, deltat):
    fig, axs = plt.subplots(1, 3, figsize=(5.3, 2.5), sharey=True)
    # ax.set_title(f"Wave Time-stepping Approximation (frame: 1)")
    functions = [
        r"$\sin(2\pi x)$",
        r"$\sin(5\pi x)$",
        r"$\sin(5\pi x) \text{ if } \frac{1}{5} < x < \frac{2}{5}, \text{ else } 0$",
    ]
    fig.suptitle("Wave Time-Stepping Animation")
    all_soltjes = []
    for j in range(3):
        soltjes, xs, deltax = solutions.initialize_wave(j + 1, L, N)
        all_soltjes.append(soltjes)
        axs[j].plot(xs, all_soltjes[j][1])
        # axs[j].set_title(functions[j])
        axs[j].set_title("i" * (j + 1))
        axs[j].set_xlabel("x")

    axs[0].set_ylabel(r"$\Psi$")

    plt.tight_layout()  # Prevent overlap
    fig.subplots_adjust(top=0.8)  # Move suptitle higher
    plt.pause(1)

    def animate(frame):
        nonlocal all_soltjes
        for i in range(3):
            soltjes = all_soltjes[i]
            axs[i].clear()  # Clear the previous frame
            axs[i].set_xlim(0, L)  # Set x-axis limits
            axs[i].set_ylim(-1, 1)  # Set y-axis limits
            # axs[i].set_title(functions[i])
            axs[i].set_title("i" * (i + 1))
            for _ in range(100):
                soltjes = solutions.wave_step_function(soltjes, c, xs, deltax, deltat)
            all_soltjes[i] = soltjes
            axs[i].plot(xs, soltjes[1])
            axs[i].set_xlabel("x")

        axs[0].set_ylabel(r"$\Psi$")

    animation = FuncAnimation(fig, animate, frames=1500, interval=1)
    animation.save("plots/network_animation10.gif", fps=50, writer="ffmpeg")
    plt.show()


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

    if os.path.exists("data/2D_diffusion.pkl"):
        all_c, times = pkl.load(open("data/2D_diffusion.pkl", "rb"))
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


def plot_simulation_without_animation(grids, N):
    fig, ax = plt.subplots(figsize=(7, 7))

    c_plot = ax.pcolormesh(grids[-1], cmap="viridis", edgecolors="k", linewidth=0.5)

    # plt.imshow(c, cmap="viridis", interpolation="nearest", origin="lower")
    plt.colorbar(c_plot, ax=ax, label="Concentration")
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("2D Diffusion")
    plt.show()


def animate_2f(update_func, grid, num_steps, N, gamma, dt, interval=50):
    path = "data/2D_diffusion_comparison.pkl"
    if os.path.exists(path):
        grids, times = pkl.load(open(path, "rb"))
    else:
        grids, times = update_func(grid, num_steps, N, gamma, dt)

    print(f"Starting grid: {grid}")

    fig, axs = plt.subplots(figsize=(6, 6))
    img = axs.imshow(grids[0], cmap="viridis", interpolation="nearest", origin="lower")
    plt.colorbar(img, ax=axs, label="Concentration")
    axs.set_title("2D Diffusion Simulation")

    def animate(frame):
        img.set_array(grids[frame])
        axs.set_title(f"2D Diffusion Simulation (Step: {frame:.3g})")

    animation = FuncAnimation(
        fig, animate, frames=len(grids), interval=interval, blit=False
    )
    animation.save("plots/2D_diffusion.gif", fps=50, writer="ffmpeg")

    plt.close(fig)
    return animation


def analytical_solution(x, t, D=1, i_max=100):
    if t <= 0:
        return np.nan

    sum_val = 0.0
    for i in range(i_max + 1):
        arg_1 = (1 - x + 2 * i) / (2 * np.sqrt(D * t))
        arg_2 = (1 + x + 2 * i) / (2 * np.sqrt(D * t))
        sum_val += erfc(arg_1) - erfc(arg_2)

    return sum_val


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
