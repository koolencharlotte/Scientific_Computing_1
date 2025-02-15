import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import src.solutions as solutions

def visualization_1b(overall_solutions, xs):
    fig, axs = plt.subplots(1, 3, figsize=(5.3,2.5), sharey=True)

    for j in range(len(overall_solutions)):
        for k in range(len(overall_solutions[j])):
            axs[j].plot(xs, overall_solutions[j][k], linewidth=1)
        axs[j].set_xlabel("x")
        axs[j].set_title('i'*(j+1))
   
    axs[0].set_ylabel(r'$\Psi^n$')

    fig.suptitle("Wave Time-stepping Approximation")
    plt.tight_layout()
    fig.subplots_adjust(top=0.8) 
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
            for _ in range(100):
                soltjes = solutions.wave_step_function(soltjes, c, xs, deltax, deltat)
            all_soltjes[i] = soltjes
            axs[i].plot(xs, soltjes[1]) 
            axs[i].set_xlabel("x")
   
        axs[0].set_ylabel(r'$\Psi$')

    
    animation = FuncAnimation(fig, animate, frames=1500, interval=1)
    animation.save("plots/network_animation10.gif", fps=50,  writer="ffmpeg")
    plt.show()

def visualization_1i(p_values, iterations_jacobi, iterations_gauss_seidel, iterations_sor):
    
    plt.figure(figsize=(5.3, 2.5))
    plt.plot(p_values, iterations_jacobi, color='blue', label="Jacobi")
    plt.plot(p_values, iterations_gauss_seidel, color='orange', label="Gauss-Seidel")
    plt.plot(p_values, iterations_sor, color='green', label="Successive Over Relaxation")
    plt.xlabel(r'$p$', fontsize=14)
    plt.ylabel('Iterations', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc="upper left")
    plt.grid(True)
    plt.title("Convergence Measure vs. Iterations")
    plt.savefig("plots/fig_1i.png", dpi=300, bbox_inches="tight")
    plt.show()

def visualization_1j_omega_iters(iters_N, omega_range):

    plt.figure(figsize=(5.3, 2.5))

    for N, iters in iters_N.items():
        plt.plot(omega_range, iters, label=f"N = {N}")

    plt.xlabel(r'$\omega$', fontsize=14)
    plt.ylabel('Iterations', fontsize=14)
    plt.legend(fontsize=10, loc="upper left")
    plt.title("SOR Convergence Speed vs. ω for Different Grid Sizes", fontsize=14)
    plt.grid(True)
    plt.savefig("plots/fig_1ja.png", dpi=300, bbox_inches="tight")
    plt.show()

def visualization_1j_N_omegas(N_values, optimal_omegas):

    plt.figure(figsize=(5.3, 2.5))
    plt.plot(N_values, optimal_omegas, marker='o', linestyle='-')
    plt.xlabel('Grid Size N', fontsize=14)
    plt.ylabel('Optimal ω', fontsize=14)
    plt.title("Optimal ω vs. Grid Size N", fontsize=14)
    plt.grid(True)
    plt.savefig("plots/fig_1jb.png", dpi=300, bbox_inches="tight")
    plt.show()