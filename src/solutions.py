import os
import pickle as pkl

import numpy as np
from math import erfc

def spat_approx_1a(deltax, solutions):
    """
    Parameters:
    deltax (float): Spatial step size.
    solutions (tuple): Three consecutive solution values (previous, current, next).

    Returns:
    float: Second-order finite difference approximation.
    """

    assert deltax != 0, "can't divide by 0 (spatial approximation)"
    return (solutions[2] - 2 * solutions[1] + solutions[0]) / np.power(deltax, 2)


def initialize_wave(which_one, L, N):
    """
    Initializes the wave function with a chosen initial condition.

    Parameters:
    which_one (int): Selects the initial condition (1, 2, or 3).
    L (float): Length of the spatial domain.
    N (int): Number of spatial divisions.

    Returns:
    tuple: Initial wave solutions (previous, current, next), spatial points, and deltax.
    """

    # Functions to initialize configuration
    def b_one(x):
        return np.sin(2 * np.pi * x)

    def b_two(x):
        return np.sin(5 * np.pi * x)

    def b_three(x):
        if x > 1 / 5 and x < 2 / 5:
            return np.sin(5 * np.pi * x)
        else:
            return 0

    # Spatial step size
    assert N > 0, "Number of subparts must be greater than 0 (zero-division)"
    deltax = L / N
    xs = np.arange(0, L, deltax)

    # Use specified function to model initial configuration
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
    # This solution is the same as previous as the derivative is 0 (applying Euler's method)
    sols = sols_prev.copy()

    # Next solutions are still empty
    sols_next = np.zeros(len(xs))

    return (sols_prev, sols, sols_next), xs, deltax


def wave_step_function(all_sols, c, xs, deltax, deltat):
    """
    Performs one time step of the wave equation using finite differencing.

    Parameters:
    all_sols (tuple): Contains previous, current, and next solution arrays.
    c (float): Wave speed.
    xs (numpy array): Spatial grid points.
    deltax (float): Spatial step size.
    deltat (float): Time step size.

    Returns:
    tuple: Updated wave solutions (previous, current, next).
    """
    sols_prev, sols, sols_next = all_sols
    for j, x in enumerate(xs):
        # Border conditions
        if j == 0:
            sols_next[j] = 0
        elif j == len(xs) - 1:
            sols_next[j] = 0

        # In case point is not a border point, update according to previous value and neighboring values
        else:
            sols_next[j] = (
                np.power(deltat, 2)
                * np.power(c, 2)
                * spat_approx_1a(deltax, (sols[j - 1], sols[j], sols[j + 1]))
                + 2 * sols[j]
                - sols_prev[j]
            )

    # Update saved data
    sols_prev = sols.copy()
    sols = sols_next.copy()

    # Return updated data
    return sols_prev, sols, sols_next


def one_b_wrapper(which_one, L, N, c, deltat, iters=20000):
    """
    Simulates wave propagation over time using numerical methods.

    Parameters:
    which_one (int): Selects the initial condition (1, 2, or 3).
    L (float): Length of the spatial domain.
    N (int): Number of spatial divisions.
    c (float): Wave speed.
    deltat (float): Time step size.
    iters (int, optional): Number of time iterations (default: 20000).

    Returns:
    tuple: List of wave solutions at selected time steps and spatial grid points.
    """
    overall_solutions = []
    every_what = int(iters / 10)

    # Initialize the configuration with a specified function (specified with which_one)
    soltjes, xs, deltax = initialize_wave(which_one, L, N)
    overall_solutions.append(soltjes[1])

    # Iterate through specified number of iterations and save 10 evenly separated lines overall
    for i in range(iters):
        # Perform a step
        soltjes = wave_step_function(soltjes, c, xs, deltax, deltat)

        # Save data for visualization purposes
        if i % every_what == 0:
            overall_solutions.append(soltjes[1])

    # Return solution and x-values for which these solutions are computed
    return overall_solutions, xs

def two_dimensional_step_wave_function(all_sols, c, xs, deltax, deltat):
    pass

def analytical_solution(x, t, D=1, i_max=100):
    """
    Function describing the analytical solution to the 2D 
    diffusion equation/ 
    """
    if t <= 0:  
        return 0

    sum_val = 0.0
    for i in range(i_max + 1):
        arg_1 = (1 - x + 2 * i) / (2 * np.sqrt(D * t))
        arg_2 = (1 + x + 2 * i) / (2 * np.sqrt(D * t))
        sum_val += erfc(arg_1) - erfc(arg_2)

    return sum_val


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


def check_and_parse_data(data_file, newdata, values):

    # check if main folder exists
    assert os.path.exists("data"), "Directory to the data folder does not exist (create directory data)"

    c, num_steps, N, gamma, dt = values
    # if existing data is used for simulation, this data is chosen

    if not newdata:
        if os.path.exists(f"data/{data_file}"):
            all_c, times = pkl.load(open(f"data/{data_file}", "rb"))
        else:
            raise ValueError(f"the data {data_file} does not exist, choose an existing file")
    else:
        all_c, times = update(c, num_steps, N, gamma, dt, comparison=True)

    return all_c, times

def sequential_jacobi(N, tol, max_iters):
    """
    Solves using the Jacobi iteration method.
    
    The update equation is:
        c_{i,j}^{k+1} = (1/4) * (c_{i+1,j}^{k} + c_{i-1,j}^{k} + c_{i,j+1}^{k} + c_{i,j-1}^{k})
    
    Parameters:
        N (int): Grid size.
        tol (float): Convergence tolerance.
        max_iters (int): Maximum number of iterations.

    Returns:
        int: Number of iterations required to reach convergence.
    """

    # grid initialisation
    c_old = initialize_grid(N)
    c_next = np.copy(c_old)

    iter = 0
    delta = float("inf")

    while delta > tol and iter < max_iters:
        delta = 0

        for i in range(1, N - 1):  # periodic in x
            for j in range(1, N - 1):  # fixed in y
                # add
                # if (c_old is a source) c_next = cL
                # else if (c_old is a sink) c_next = c0

                # periodic boundaries
                if i == 0:
                    west = c_old[N - 1, j]
                else:
                    west = c_old[i - 1, j]

                if i == N - 1:
                    east = c_old[0, j]
                else:
                    east = c_old[i + 1, j]

                # fixed boundaries
                south = 0 if j == 0 else c_old[i, j - 1]
                north = 1 if j == N - 1 else c_old[i, j + 1]

                # Jacobi update equation
                c_next[i, j] = 0.25 * (west + east + south + north)

                delta = max(delta, abs(c_next[i, j] - c_old[i, j]))

        # swap matrices for next iter
        c_old[:], c_next[:] = c_next, c_old

        iter += 1

    return iter


# sequential gauss seidel
def sequential_gauss_seidel(N, tol, max_iters):
    """
    Solves using the Gauss-Seidel iteration method.
    
    The update equation is:
        c_{i,j}^{n+1} = (1/4) * (c_{i+1,j}^{k} + c_{i-1,j}^{k+1} + c_{i,j+1}^{k} + c_{i,j-1}^{k+1})
    
    Parameters:
        N (int): Grid size.
        tol (float): Convergence tolerance.
        max_iters (int): Maximum number of iterations.

    Returns:
        int: Number of iterations required to reach convergence.
    """

    # grid initialisation
    c = initialize_grid(N)

    iter = 0
    delta = float("inf")

    while delta > tol and iter < max_iters:
        delta = 0

        for i in range(1, N - 1):  # periodic in x
            for j in range(1, N - 1):  # fixed in y
                # periodic boundary conditions
                west = c[i - 1, j] if i > 0 else c[N - 1, j]
                east = c[i + 1, j] if i < N - 1 else c[0, j]
                south = c[i, j - 1] if j > 0 else c0
                north = c[i, j + 1] if j < N - 1 else cL

                # Gauss-Seidel update equation
                c_next = 0.25 * (west + east + south + north)

                delta = max(delta, abs(c_next - c[i, j]))
                c[i, j] = c_next

        iter += 1

    return iter

# sequential SOR iteration
def sequential_SOR(N, tol, max_iters, omega):
    """
    Solves using the Successive Over Relaxtion (SOR) iteration method.
    
    The update equation is:
        c_{i,j}^{k+1} = (omega/4) * (c_{i+1,j}^{k} + c_{i,j+1}^{k} + c_{i,j+1}^{k} + (1 - omega) c_{i,j}^{k})
    
    Parameters:
        N (int): Grid size.
        tol (float): Convergence tolerance.
        max_iters (int): Maximum number of iterations.
        omega (float): Relaxation factor.

    Returns:
        int: Number of iterations required to reach convergence.
    """

    # grid initialisation
    c = initialize_grid(N)

    iter = 0
    delta = float('inf')

    while delta > tol and iter < max_iters:
        delta = 0

        for i in range(1, N-1):  # periodic in x
            for j in range(1, N-1):  # fixed in y

                # periodic boundary conditions
                west = c[i - 1, j] if i > 0 else c[N - 1, j]
                east = c[i + 1, j] if i < N - 1 else c[0, j]
                south = c[i, j - 1] if j > 0 else 0
                north = c[i, j + 1] if j < N - 1 else 1

                # SOR update equation
                c_next = (omega / 4) * (west + east + south + north) + (1 - omega) * c[i, j]

                delta = max(delta, abs(c_next - c[i, j]))
                c[i, j] = c_next

        iter += 1

    return iter
