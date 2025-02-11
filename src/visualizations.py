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