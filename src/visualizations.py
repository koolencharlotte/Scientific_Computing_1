import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import src.solutions as solutions
from matplotlib.animation import FuncAnimation

from .solutions import initialize_grid, update


def visualization_1b(overall_solutions, xs):
    fig, axs = plt.subplots(1, 3, figsize=(5.3,2.5), sharey=True)

    for j in range(len(overall_solutions)):
        for k in range(len(overall_solutions[j])):
            axs[j].plot(xs, overall_solutions[j][k], linewidth=0.9)
        axs[j].set_xlabel("x")
        axs[j].set_title('i'*(j+1))
   
    axs[0].set_ylabel(r'$\Psi^n$', labelpad=1)
    axs[0].yaxis.set_label_coords(-0.22, 0.5)

    fig.suptitle("Wave Time-stepping Approximation")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08, top=0.8) 
    plt.savefig("plots/fig_1B.png", dpi=300)
    plt.show()


def animate_1c(L, N, c, deltat):

    fig, axs = plt.subplots(1, 3, figsize=(5.3,2.5), sharey=True)
    # ax.set_title(f"Wave Time-stepping Approximation (frame: 1)")
    functions = [
    r"$\sin(2\pi x)$",
    r"$\sin(5\pi x)$",
    r"$\sin(5\pi x) \text{ if } \frac{1}{5} < x < \frac{2}{5}, \text{ else } 0$"
    ]
    fig.suptitle("Wave Time-Stepping Animation")
    all_soltjes = []
    for j in range(3):
        soltjes, xs, deltax = solutions.initialize_wave(j+1, L, N)
        all_soltjes.append(soltjes)
        axs[j].plot(xs, all_soltjes[j][1]) 
        # axs[j].set_title(functions[j])
        axs[j].set_title('i'*(j+1))
        axs[j].set_xlabel("x")
   
    axs[0].set_ylabel(r'$\Psi$')
    
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
            axs[i].set_title('i'*(i+1))
            for _ in range(10):
                soltjes = solutions.wave_step_function(soltjes, c, xs, deltax, deltat)
            all_soltjes[i] = soltjes
            axs[i].plot(xs, soltjes[1]) 
            axs[i].set_xlabel("x")
   
        axs[0].set_ylabel(r'$\Psi$')

    
    animation = FuncAnimation(fig, animate, frames=300, interval=1)
    animation.save("plots/network_animation10.gif", fps=50,  writer="ffmpeg")
    plt.show()



def plot_analytical_solution(y, solutions_vals, times, D):
    """
    Funcion comparing analytical solution to numerical solution. 
    Plotting points for numerical solution and line for analytical soltution 
    """
    nx = 100
    y_analytic = np.linspace(0.0, 1.0, nx)

    plt.figure(figsize=(4.5, 3))
    colors = ["orange", "blue", "green", "purple", "brown"]

    for i, t_val in enumerate(times):
        # Analytical
        c_analytical = [solutions.analytical_solution(yy, t_val, D=D) for yy in y_analytic]
        plt.plot(y_analytic, c_analytical, color = colors[i], linewidth=0.8, label=f"t={t_val:.3g}")

        c_analytical = np.array(c_analytical)

        # mean of the row (should theoretically all be the same)
        c2D = solutions_vals[i]
        # c_avg = np.mean(c2D, axis=1)
        c_first = c2D[:, 0]  

        # slice array to avoid clutteredness
        plt.plot(y[::2], c_first[::2], "o", color = colors[i], markersize=2)

    plt.xlabel("y (Position)")
    plt.ylabel("Concentration c(y, t)")
    plt.legend()
    plt.title("Diffusion Simulation vs Analytical Solution")
    plt.grid(True)
    plt.savefig("plots/diffusion_analytical.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_five_states(all_c, times):
    """
    function plotting the 2D diffustion grid, 
    5 states for t = 0, 0.001, 0.01, 0.1, 1.0
    """

    # # read in/ generate data 
    # data_file = "2D_diffusion.pkl"
    # create_new_data = False
    # all_c, times = solutions.check_and_parse_data(data_file, create_new_data, parameters)

    # plot setup
    fig, axs = plt.subplots(2, 3, figsize=(5.8, 4.3), sharex=True, sharey=True)
    fig.suptitle("2D Diffusion at Different t Values")
    axs = axs.flatten()  

    # Hide the last unused subplot
    axs[-1].set_visible(False)  
    for i in range(5):
        im = axs[i].imshow(all_c[i], cmap="viridis", interpolation="nearest", origin="lower")
        axs[i].set_title(f"t = {times[i]}")
        if i >1: 
            axs[i].set_xlabel("x")

    # set proper ticks and labels
    axs[2].xaxis.set_tick_params(which='both', labelbottom=True)
    axs[0].set_ylabel("y")
    axs[3].set_ylabel("y")

    cbar_ax = fig.add_axes([0.92, 0.09, 0.02, 0.75])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label="Concentration")

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.subplots_adjust(wspace=0.05, hspace=0.2) 
    plt.savefig("plots/diffusion_snapshots.png", dpi=300)
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


