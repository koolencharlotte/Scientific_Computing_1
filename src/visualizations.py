import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import src.solutions as solutions
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors 
import matplotlib.cm as cm
# 1B

def visualization_1b(overall_solutions, xs, iterations, deltat):
    """
    Plots wave time-stepping approximations with color-coded lines indicating time progression.

    Parameters:
        overall_solutions (list of lists): Contains solution values for each timestep.
        xs (ndarray): Spatial grid points.
        iterations (int): Number of timesteps.
        deltat (float): Time step size.
    """

    max_time = iterations * deltat
    num_lines = len(overall_solutions[0])  

    # Normalize time values for colormap
    norm = mcolors.Normalize(vmin=0, vmax=max_time)
    cmap = cm.viridis # Choose a colormap
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)  # For colorbar

    fig, axs = plt.subplots(1, 3, figsize=(4.8, 2.8), sharey=True)

    for j in range(len(overall_solutions)):
        for k in range(num_lines):
            time_val = k * (max_time / num_lines)  # Assign time to each line
            axs[j].plot(xs, overall_solutions[j][k], linewidth=0.9, color=cmap(norm(time_val)))

        axs[j].set_xlabel("x")
        axs[j].set_title("i" * (j + 1))

    axs[0].set_ylabel(r"$\Psi^n$", labelpad=1)
    axs[0].yaxis.set_label_coords(-0.22, 0.5)

    # Adding horizontal colorbar below the subplots
    cbar_ax = fig.add_axes([0.13, 0.07, 0.82, 0.03]) 
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Time (S)")

    fig.suptitle("Wave Time-stepping Approximation")
    plt.tight_layout() 
    plt.subplots_adjust(wspace=0.08, top=0.8, bottom=0.27)  

    plt.savefig("plots/fig_1B.png", dpi=300)
    plt.show()

#1C
def animate_1c(L, N, c, deltat):
    """
    Generates an animated visualization of wave propagation using three different 
    initial wave functions. The animation updates the wave solutions over time 
    based on a finite difference time-stepping method.

    Parameters:
        L  (float): Length of the spatial domain.
        N (int): Number of spatial grid points.
        c (float): Wave speed.
        deltat (float):Time step size.

    """ 

    fig, axs = plt.subplots(1, 3, figsize=(5.3,2.5), sharey=True)

    fig.suptitle("Wave Time-Stepping Animation")
    all_soltjes = []

    # iterate over the three functions and initialize wave
    for j in range(3):
        soltjes, xs, deltax = solutions.initialize_wave(j + 1, L, N)

        # compute solutions of initial wave and save them for visualization
        all_soltjes.append(soltjes)
        axs[j].plot(xs, all_soltjes[j][1]) 
        axs[j].set_title('i'*(j+1))
        axs[j].set_xlabel("x")

    axs[0].set_ylabel(r"$\Psi$")

    plt.tight_layout()  
    fig.subplots_adjust(top=0.8)  

    # pause to visualize first frame
    plt.pause(1)

    def animate(frame):
        nonlocal all_soltjes

        # for every inital function
        for i in range(3):
            
            # retrieve previous solution for this function
            soltjes = all_soltjes[i]

            # configure layout for each subplot
            axs[i].clear()  
            axs[i].set_xlim(0, L)  
            axs[i].set_ylim(-1, 1) 
            axs[i].set_title("i" * (i + 1))

            # iterate over 10 iterations before visualization (this has no visual effect)
            for _ in range(10):

                # compute solutions for this timestep
                soltjes = solutions.wave_step_function(soltjes, c, xs, deltax, deltat)

            # plot solution
            all_soltjes[i] = soltjes
            axs[i].plot(xs, soltjes[1])
            axs[i].set_xlabel("x")

        axs[0].set_ylabel(r"$\Psi$")

    # generate animation
    animation = FuncAnimation(fig, animate, frames=300, interval=1)
    animation.save("plots/network_animation10.gif", fps=50, writer="ffmpeg")
    plt.close(fig)

    return animation


def plot_analytical_solution_with_error(y, solution_vals, times, D):
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
        c_analytical = [
            solutions.analytical_solution(yy, t_val, D=D) for yy in y_analytic
        ]
        ax1.plot(
            y_analytic,
            c_analytical,
            color=colors[i],
            linewidth=0.8,
            label=f"t={t_val:.3g}",
        )

        c_analytical = np.array(c_analytical)

        # mean of the row (should theoretically all be the same)
        c2D = solution_vals[i]
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
    fig1.savefig("plots/diffusion_analytical.png", dpi=300, bbox_inches="tight")

    ax2.set_xlabel("y (Position)")
    ax2.set_ylabel("Absolute error")
    ax2.legend()
    ax2.set_ylim(0, 0.008)
    ax2.set_title("Error Simulation and Analytical Solution")
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig("plots/diffusion_analytical_error.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_five_states(all_c, times):
    """
    function plotting the 2D diffustion grid,
    5 states for t = 0, 0.001, 0.01, 0.1, 1.0
    """

    # plot setup
    fig, axs = plt.subplots(2, 3, figsize=(4.4, 3.4), sharex=True, sharey=True)
    fig.suptitle("2D Diffusion at Different t Values")
    axs = axs.flatten()

    # Hide the last unused subplot
    axs[-1].set_visible(False)
    for i in range(5):
        im = axs[i].imshow(
            all_c[i], cmap="viridis", interpolation="nearest", origin="lower", extent=[0, 1, 0, 1]
        )
        axs[i].set_title(f"t = {times[i]}")
        if i > 1:
            axs[i].set_xlabel("x")

    # set proper ticks and labels
    axs[2].xaxis.set_tick_params(which="both", labelbottom=True)
    axs[0].set_ylabel("y")
    axs[3].set_ylabel("y")

    cbar_ax = fig.add_axes([0.92, 0.09, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label="Concentration")

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.subplots_adjust(wspace=0.22, hspace=0.2)
    plt.savefig("plots/diffusion_snapshots.png", dpi=300, bbox_inches="tight")
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

    # print(f"Starting grid: {grid}")

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

def visualization_1h(c_jacobi_2d, c_gs_2d, c_sor_2d, N):
    """
    Visualizes the absolute error betweeen the solution of the iterative methods: Jacobi, Gauss-Seidel,
    and SOR methods with the analytical solution for a grid size of 50.
    
    Parameters:
        c_jacobi_2d: 2D array representing the solution using the Jacobi method.
        c_gs_2d: 2D array representing the solution using the Gauss-Seidel method.
        c_sor_2d: 2D array representing the solution using the SOR method.
        N (int): The number of grid points used in the simulation.
    """

    y_analytic = np.linspace(0, 1, N)

    c_jacobi_1d = c_jacobi_2d[:,0]
    c_gs_1d = c_gs_2d[:,0]
    c_sor_1d = c_sor_2d[:,0]

    error_jacobi = np.abs(c_jacobi_1d - y_analytic)
    error_gs = np.abs(c_gs_1d - y_analytic)
    error_sor = np.abs(c_sor_1d - y_analytic)

    
    plt.figure(figsize=(6.7, 3.5))
    plt.plot(y_analytic, error_jacobi, "o", markersize=2, label="Jacobi")
    plt.plot(y_analytic, error_gs, "o", markersize=2, label="Gauss-Seidel")
    plt.plot(y_analytic, error_sor, "o", markersize=2, label="SOR")
    plt.xlabel("y (Position)")
    plt.ylabel("Absolute error")
    plt.title("Error Simulation and Analytical Solution")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/fig_1h.png", dpi=300, bbox_inches="tight")
    plt.show()

def visualization_1i(
    p_values, iterations_jacobi, iterations_gauss_seidel, iterations_sor, colors
):
    """
    Visualizes the convergence measure vs. the number of iterations for Jacobi, Gauss-Seidel,
    and SOR methods across different values of p.

    Parameters:
        p_values (list): The p-values to be plotted on the x-axis.
        iterations_jacobi (list): Number of iterations for the Jacobi method.
        iterations_gauss_seidel (list): Number of iterations for the Gauss-Seidel method.
        iterations_sor (dict): Number of iterations for different values of omega in the SOR method.
        colors (list): List of colors to use for each method's plot.
    """

    plt.figure(figsize=(5.3, 2.5))

    linestyles = ["dotted", "dashed", "dashdot", "solid"]
    num_styles = len(linestyles)

    plt.plot(p_values, iterations_jacobi, color=colors[0], label="Jacobi")
    plt.plot(p_values, iterations_gauss_seidel, color=colors[1], label="Gauss-Seidel")

    for i, (omega, sor_iterations) in enumerate(iterations_sor.items()):
        plt.plot(
            p_values,
            sor_iterations,
            label=f"SOR (ω={omega})",
            color=colors[2],
            linestyle=linestyles[i % num_styles],
        )

    plt.xlabel(r"$p$", fontsize=14)
    plt.ylabel("Iterations", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc="upper left")
    plt.grid(True)
    plt.title("Convergence Measure vs Iterations")
    plt.savefig("plots/fig_1i.png", dpi=300, bbox_inches="tight")
    plt.show()


def visualization_1j_omega_iters(iters_N, omega_range, colors):
    """
    Visualizes the convergence speed of the SOR method across different omega values for varying grid sizes.

    Parameters:
        iters_N (dict): A dictionary where keys are grid sizes (N) and values are lists of iterations.
        omega_range (list): List of omega values used for the SOR method.
        colors (list): List of colors to use for each method's plot.
    """

    plt.figure(figsize=(5.3, 3.6))

    linestyles = ["dotted", "dashed", "dashdot", "solid"]
    num_styles = len(linestyles)

    for i, (N, iters) in enumerate(iters_N.items()):
        plt.plot(
            omega_range,
            iters,
            label=f"N = {N}",
            color=colors[2],
            linestyle=linestyles[i % num_styles],
        )

    plt.xlabel(r"$\omega$", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Iterations", fontsize=14)
    plt.legend(fontsize=10, loc="upper left", ncol=2)
    plt.title("SOR Convergence Speed vs ω", fontsize=14)
    plt.grid(True)
    plt.savefig("plots/fig_1ja.png", dpi=300, bbox_inches="tight")
    plt.show()


def visualization_1j_N_omegas(N_values, optimal_omegas, colors):
    """
    Visualizes the relationship between the optimal omega and grid size N for the SOR method.

    Parameters:
        N_values (list): List of grid sizes (N) plotted on the x-axis.
        optimal_omegas (list): List of optimal omega values for each grid size.
        colors (list): List of colors to use for the plot.
    """

    plt.figure(figsize=(5.3, 2.5))
    plt.plot(N_values, optimal_omegas, marker="o", linestyle="-", color=colors[2])
    plt.xlabel("Grid Size N", fontsize=14)
    plt.ylabel("Optimal ω", fontsize=14)
    plt.title("Optimal ω vs Grid Size N", fontsize=14)
    plt.grid(True)
    plt.savefig("plots/fig_1jb.png", dpi=300, bbox_inches="tight")
    plt.show()

# 1K
def visualize_object_grid(obj_grids, sizes):
    """
    Visualizes a 2x2 grid of object configurations using binary occupancy maps.

    Parameters:
        obj_grids (list of list of ndarray): A list containing four object grid configurations (2D).
        sizes (list of str): A list of four labels corresponding to the object configurations.
    """
    fig, axs = plt.subplots(2, 2, figsize=(3.1, 3.8))

    # grey for spaces not occupied and black for occupied spaces
    cmap = mcolors.ListedColormap(["lightgrey", "black"])  
    axs = axs.flatten()

    # visualize grid for every object configuraiton
    for i in range(4):
        axs[i].imshow(obj_grids[i][0], cmap=cmap)  
        axs[i].set_xticks([])   
        axs[i].set_yticks([])  
        axs[i].grid(False)  
        axs[i].set_title(sizes[i])
    
    plt.suptitle(f"Object Grids ({len(obj_grids[0][0])}x{len(obj_grids[0][1])})")
    plt.tight_layout()
    plt.savefig("plots/object_layout.png", dpi=300)
    plt.show() 


def vis_object_per_gridsize(all_grids, all_grids_omega, null_measure, null_measure_omega, config_labels, sizes, colors):
    """
    Visualizes the convergence of an object grid for different grid sizes and omega values.

    Generates a figure with two subplots:
    - The first shows mean iterations vs. grid size.
    - The second shows mean iterations vs. omega values.
    Variance is displayed as a shaded region.

    Parameters:
        all_grids, all_grids_omega (dict): Mapping of grid sizes and omega values to mean/variance data.
        null_measure, null_measure_omega (list): Reference values plotted as dashed lines.
        config_labels (list of str): Labels for different object configurations.
        sizes (list of str): Legend labels for configurations.
        colors (list of str): Colors for each configuration.
    """
   
    fig, axs = plt.subplots(2, 1, figsize=(4, 5.5))

    # incase iterated through omega: grid represent different omega values
    grids = sorted(all_grids.keys())  
    omegas = sorted(all_grids_omega.keys())
    grid_om = [all_grids, all_grids_omega]

    keyvals = [grids, omegas]
    # Iterate over each object configuration and plot a separate line
    for i, config_label in enumerate(config_labels):

        for j in range(2):
            means = []
            vartjes = []
            
            # loop over the different experiments and access mean and variance
            for grid_size in keyvals[j]:
                config_data = grid_om[j][grid_size]  # Get the dictionary for this grid size
                means.append(config_data[config_label][0])  # Mean value
                vartjes.append(config_data[config_label][1])  # Variance value
    
            # Plot line for this configuration
            axs[j].plot(keyvals[j], means, label=f"{sizes[i]}", marker="o", color=colors[i])

            # Plot variance as a shaded region
            axs[j].fill_between(keyvals[j], 
                            np.array(means) - np.array(np.sqrt(vartjes)), 
                            np.array(means) + np.array(np.sqrt(vartjes)), 
                            alpha=0.2, color=colors[i])

    # Plot null measure for reference
    axs[0].plot(grids, null_measure, '--', label=f"{sizes[-1]}", color="black")
    axs[1].plot(omegas, null_measure_omega, '--', label=f"{sizes[-1]}", color="black")
    axs[0].set_xlabel("Grid Size")
    axs[1].set_xlabel(r"$\omega$")
    
    axs[0].set_title(r"For Different Grid-Sizes ($\omega$ = 1.9)" )
    axs[1].set_title(r"For Different $\omega$ Values (50x50)")

    # Labels and legend

    axs[0].set_ylabel("Iterations")
    axs[1].set_ylabel("Iterations")

    axs[0].legend()
    plt.suptitle("Convergence on Object Grid", fontsize= 14)
    plt.tight_layout()
    plt.savefig("plots/questionK.png", dpi=300)
    plt.show()


def plot_converged_object_grid(all_c, iterate_through,config, which_one="N"):
    """
    Plots converged object grids for different experimental conditions.

    Parameters:
        all_c (dict): Dictionary mapping experiment values to diffusion grids.
        iterate_through (list): Values over which the experiments are conducted (grid sizes or omega values).
        config (int): Index of the object configuration to visualize.
        which_one (str, optional): Specifies the experiment type:
            - "N": Varying grid size (default).
            - "O": Varying relaxation factor ω.
   """

    assert len(all_c.keys()) == len(iterate_through), "Data provided and variables used for experimentation don't allign"

    # plot setup
    if which_one == "O":
        sizeje = (6.6, 5.3)
        fx = 3
        fy = 2
        titletje = r"$\omega$"
        h=0.78
    elif which_one == "N":
        sizeje= (6.6, 3.0)
        fx = 3
        fy =1
        titletje = "Grid Size"
        h = 0.65
    else: 
        raise ValueError(f"{which_one} is not a valid experimentation, choose N or O")
    
    fig, axs = plt.subplots(fy, fx, figsize=sizeje)
    fig.suptitle(f"Grid diffusion (Object configuration {config})")
    axs = axs.flatten()

    for ktje, i in enumerate(iterate_through):
        # print(all_c[i])
        im = axs[ktje].imshow(
            all_c[i][config], cmap="viridis", interpolation="nearest", origin="lower"
        )
        if fx <2:
            axs[ktje].set_xlabel("x")
        elif ktje > 2:
            axs[ktje].set_xlabel("x")
        axs[ktje].set_title(titletje + f"{i}")
  
    axs[0].set_ylabel("y")
    if fy > 1:
        axs[3].set_ylabel("y")

    cbar_ax = fig.add_axes([0.92, 0.09, 0.02, h])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label="Concentration")

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig("plots/diffusion_snapshots.png", dpi=300)
    plt.show()