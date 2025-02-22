import os
import pickle as pkl
from math import erfc
from multiprocessing import Pool

import numpy as np
from numba import njit, prange
from scipy.stats import ttest_ind_from_stats


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


def analytical_solution(x, t, D=1, i_max=100):
    """
    Function describing the analytical solution to the 2D
    diffusion equation/
    """
    if t <= 0:
        return 0

    sum_val = 0.0
    # formula for the analytical solution as given in the assignment description
    for i in range(i_max + 1):
        arg_1 = (1 - x + 2 * i) / (2 * np.sqrt(D * t))
        arg_2 = (1 + x + 2 * i) / (2 * np.sqrt(D * t))
        sum_val += erfc(arg_1) - erfc(arg_2)

    return sum_val


def initialize_grid(N):
    """
    Generates a grid with the specified dimensions and initializes the boundaries.
    Parameters:
        N (int): Grid size.
    """
    assert N > 1, "Grid size must be bigger than 1x1"
    grid = np.zeros((N, N))

    grid[0, :] = 0  # bottom boundary
    grid[N - 1, :] = 1  # top boundary

    return grid


def apply_periodic_boundary(grid):
    """
    Applies periodic boundary conditions to the grid in horizontal direction.
    """

    grid[:, 0] = grid[:, -2]
    grid[:, -1] = grid[:, 1]


@njit(parallel=True)
def update(grid, num_steps, N, gamma, dt, comparison=False):
    """
    Evolves a 2D grid using an explicit finite difference scheme to simulate diffusion.

    The update follows a finite difference approximation of the 2D diffusion equation
    with periodic boundary conditions. Grid snapshots are saved at selected time steps.

    Parameters:
        grid (numpy.ndarray): Initial 2D concentration distribution.
        num_steps (int): Total number of time steps to simulate.
        N (int): Grid size (assumes an N x N grid).
        gamma (float): Diffusion coefficient factor.
        dt (float): Time increment per simulation step.
        comparison (bool, optional):
            If True, saves grid snapshots at key times (0.001, 0.01, 0.1, 1.0).
            Otherwise, saves snapshots every 100 steps. Default is False.

    Returns:
        list of numpy.ndarray: Grid snapshots at selected time steps.
        list of float: Corresponding simulation times.

    Side Effects:
        Writes a pickle file containing (all_grids, times) to disk.
        - "data/2D_diffusion_comparison.pkl" if `comparison` is True.
        - "data/2D_diffusion.pkl" otherwise.
    """

    all_grids = [grid.copy()]
    t = 0
    times = [0]

    time_appended = set()

    # the times we're experimenting with
    key_times = {0.001, 0.01, 0.1, 1.0}

    # iterate through all the steps
    for n in range(num_steps):
        c_new = grid.copy()

        # iterate through over grid (in parallel)
        for i in prange(1, N - 1):
            for j in range(N):
                south = grid[i - 1, j] if i > 0 else 0
                north = grid[i + 1, j] if i < N - 1 else 1
                west = grid[i, j - 1] if j > 0 else grid[i, N - 1]
                east = grid[i, j + 1] if j < N - 1 else grid[i, 0]

                # do the update of the cell
                c_new[i, j] = grid[i, j] + gamma * (
                    north + south + east + west - 4 * grid[i, j]
                )

        # makes sure the boundary wraps around
        # apply_periodic_boundary(c_new)

        # update old grid
        grid[:] = c_new[:]

        t = round(t + dt, 6)

        if comparison:
            # Save grid at key times for comparison against analytical solution
            for key_t in key_times:
                if np.isclose(t, key_t, atol=1e-9) and t not in time_appended:
                    all_grids.append(c_new.copy())
                    times.append(t)
                    time_appended.add(t)
        else:
            # Save grid every 100 steps for animation
            if n % 100 == 0:
                all_grids.append(grid.copy())
                times.append(t)

    return all_grids, times


def run_simulation_without_animation():
    """
    Run a 2D diffusion simulation without animation.

    This function initializes a 2D grid and performs a diffusion simulation over a total
    time of T_total = 1.0 using an explicit finite difference scheme. The simulation is set up
    with a grid size of N = 100, a spatial domain of length L = 1.0, and a diffusion coefficient D = 1.
    The time step is computed as dt = 0.25 * (dx)**2 where dx = L / N, and the diffusion factor is
    given by gamma = (D * dt) / (dx**2).

    The simulation attempts to load previously computed data from the file
    "Scientific_Computing_1/data/2D_diffusion.pkl". If the file exists, the grid snapshots and
    corresponding times are loaded from the file; otherwise, the simulation is executed by calling
    the `update` function.

    Returns
    -------
    all_grids : list of numpy.ndarray
        A list containing copies of the grid at selected time steps.
    times : list of float
        A list of simulation times corresponding to each saved grid snapshot.
    """

    N = 100
    c = initialize_grid(N)
    L = 1.0
    D = 1
    # Define initial grid
    all_grids = [c]

    dx = L / N
    dt = 0.25 * dx**2

    # Set time as 1.0 and compute number of intermediate time steps based on 'dt'
    T_total = 1.0
    num_steps = T_total / dt

    times = [0]

    # Define gamma as a separate variable to make calculations easier in the update function
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


def save_grids(all_grids, times, data_file, comparison):
    """
    Create pickle file containing grid snapshots and corresponding times.
    """
    path = f"data/comparison_{data_file}" if comparison else f"data/{data_file}"
    with open(path, "wb") as f:
        pkl.dump((all_grids, times), f)


def check_and_parse_data(data_file, newdata, values, comp):
    """
    Loads simulation data from a file or generates new data by running a simulation.

    The function checks for the existence of a "data" directory, unpacks simulation parameters
    from `values`, and either loads existing data from `data_file` or runs a new simulation using
    the `update` function with `comparison=True`.

    Parameters:
        values (tuple): Simulation parameters in the order:
            - c (numpy.ndarray): Initial grid configuration.
            - num_steps (int): Number of simulation steps.
            - N (int): Grid size (assumes an N x N grid).
            - gamma (float): Diffusion coefficient factor.
            - dt (float): Time step size.
        newdata (bool): If True, generates new data; otherwise, loads from `data_file`.
        data_file (str): Path to the simulation data file.

    Returns:
        list of numpy.ndarray: Grid snapshots from the simulation.
        list of float: Time points corresponding to each snapshot.
    """

    # check if main folder exists
    assert os.path.exists("data"), (
        "Directory to the data folder does not exist (create directory data)"
    )

    c, num_steps, N, gamma, dt = values
    # if existing data is used for simulation, this data is chosen

    if not newdata:
        if os.path.exists(f"data/{data_file}"):
            if comp: 
                all_c, times = pkl.load(open(f"data/comparison_{data_file}", "rb"))
            else:
                all_c, times = pkl.load(open(f"data/{data_file}", "rb"))
        else:
            raise ValueError(
                f"the data {data_file} does not exist, choose an existing file"
            )
    # create new data
    else:
        all_c, times = update(c, num_steps, N, gamma, dt, comp)
        save_grids(all_c, times, data_file, comp)

    return all_c, times


# sequential jacobi iteration
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
        numpy.ndarray: Final grid after iterations.
    """
    assert N > 1, (
        f"bord is {N}x{N}, but needs to be at least 2*2 for this diffusion implementation"
    )

    # grid initialisation
    c_old = initialize_grid(N)
    c_next = np.copy(c_old)

    iter = 0
    delta = float("inf")

    while delta > tol and iter < max_iters:
        delta = 0

        for i in range(1, N - 1):
            for j in range(0, N):
                south = c_old[i - 1, j] if i > 0 else 0
                north = c_old[i + 1, j] if i < N - 1 else 1
                west = c_old[i, j - 1] if j > 0 else c_old[i, N - 1]
                east = c_old[i, j + 1] if j < N - 1 else c_old[i, 0]

                # Jacobi update equation
                c_next[i, j] = 0.25 * (west + east + south + north)

                delta = max(delta, abs(c_next[i, j] - c_old[i, j]))

        # swap matrices for next iter
        c_old = np.copy(c_next)
        c_next = initialize_grid(N)

        iter += 1

    return iter, c_old


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
        numpy.ndarray: Final grid after iterations.
    """

    # grid initialisation
    assert N > 1, (
        f"bord is {N}x{N}, but needs to be at least 2*2 for this diffusion implementation"
    )
    c = initialize_grid(N)

    iter = 0
    delta = float("inf")

    # while not converged
    while delta > tol and iter < max_iters:
        delta = 0

        # loop over values in the grid (except for y=0, y=N)
        for i in range(1, N - 1):
            for j in range(0, N):
                # retrieve all necessary values (also regarding wrap-around)
                south = c[i - 1, j] if i > 0 else 0
                north = c[i + 1, j] if i < N - 1 else 1
                west = c[i, j - 1] if j > 0 else c[i, N - 1]
                east = c[i, j + 1] if j < N - 1 else c[i, 0]

                # Gauss-Seidel update equation
                c_next = 0.25 * (west + east + south + north)

                # check for convergence
                delta = max(delta, abs(c_next - c[i, j]))
                c[i, j] = c_next

        iter += 1

    return iter, c


def sequential_SOR(N, tol, max_iters, omega, object_grid=None):
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
        numpy.ndarray: Final grid after iterations.
    """

    assert N > 1, (
        f"bord is {N}x{N}, but needs to be at least 2*2 for this diffusion implementation"
    )

    # grid initialisation
    c = initialize_grid(N)

    iter = 0
    delta = float("inf")

    # while not converged
    while delta > tol and iter < max_iters:
        delta = 0

        # loop over all cells in the grid (except for y = 0, y=N)
        for i in range(1, N - 1):
            for j in range(N):
                # retrieve all necessary values (also regarding wrap-around)
                south = c[i - 1, j] if i > 0 else 0
                north = c[i + 1, j] if i < N - 1 else 1
                west = c[i, j - 1] if j > 0 else c[i, N - 1]
                east = c[i, j + 1] if j < N - 1 else c[i, 0]

                # SOR update equation
                c_next = (omega / 4) * (west + east + south + north) + (1 - omega) * c[
                    i, j
                ]

                # check for convergence
                delta = max(delta, abs(c_next - c[i, j]))
                c[i, j] = c_next

        iter += 1

    return iter, c


def non_sequential_SOR(params):
    """
    Solves using the Successive Over Relaxtion (SOR) iteration method (suitable for parallel implementation)

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
    N, tol, max_iters, omega, object_grid = params
    assert N > 1, (
        f"bord is {N}x{N}, but needs to be at least 2*2 for this diffusion implementation"
    )
    # grid initialisation
    c = initialize_grid(N)

    iter = 0
    delta = float("inf")

    # while not converged
    while delta > tol and iter < max_iters:
        delta = 0

        # loop over grid (except for y=0, y=N)
        for i in range(1, N - 1):  # periodic in x
            for j in range(N):  # fixed in y
                # if an grid point lies within the object
                if object_grid is not None and object_grid[(i, j)]:
                    c_next = 0
                    continue

                # periodic boundary conditions
                south = c[i - 1, j] if i > 0 else 0
                north = c[i + 1, j] if i < N - 1 else 1
                west = c[i, j - 1] if j > 0 else c[i, N - 1]
                east = c[i, j + 1] if j < N - 1 else c[i, 0]

                # SOR update equation
                c_next = (omega / 4) * (west + east + south + north) + (1 - omega) * c[
                    i, j
                ]

                # check for convergence
                delta = max(delta, abs(c_next - c[i, j]))
                c[i, j] = c_next

        iter += 1

    return iter, c


def place_objects(N, num_object, seed=31, size_object=4):
    """
    Randomly places square objects on an NxN grid.

    Parameters:
        N (int): Grid size (N × N).
        num_object (int): Number of objects to place.
        seed (int, optional): Random seed for reproducibility (default=31).
        size_object (int, optional): Side length of each square object (default=4).

    Returns:
        numpy.ndarray: NxN grid with placed objects, where occupied cells are marked as 1.
    """

    object_grid = np.zeros((N, N))
    np.random.seed(seed)

    # loop over number of objects
    for _ in range(num_object):
        # staying within range of the grid, + not occupying border cell -> border conditions
        x, y = np.random.randint(1, N - size_object, size=2)
        points = [
            (x + j, y + k) for j in range(size_object) for k in range(size_object)
        ]

        # set value of indexes that object occupies 1
        object_grid[tuple(zip(*points))] = 1

    return object_grid


def create_object_layouts(N, object_configs, num_grids=10):
    """
    Create grids with different object layouts
    """
    object_gridjes = []
    for num_objects, size in object_configs:
        seedje = 30
        object_grids_sub = []
        for _ in range(num_grids):
            object_grids_sub.append(place_objects(N, num_objects, seedje, size))
            seedje += 1
        object_gridjes.append(object_grids_sub)
    return object_gridjes


def generate_grid_results(
    varying,
    N,
    all_grids,
    num_grids,
    max_iters,
    omegatje,
    tol,
    object_configs,
    what_value="N",
    PROCESSES=10,
):
    """
    Runs the Successive Over-Relaxation (SOR) method in parallel for different grid sizes or omega values.

    Parameters:
        varying (list): Values over which to iterate (grid sizes or omega values).
        N (int): Default grid size (overridden if varying `N`).
        all_grids (dict): Mapping of grid sizes to object configurations and initial conditions.
        num_grids (int): Number of grids per configuration.
        max_iters (int): Maximum number of iterations allowed.
        omegatje (float): Default omega relaxation factor (overridden if varying `O`).
        tol (float): Convergence tolerance.
        object_configs (dict): Different object layouts on the grid.
        what_value (str, optional): Determines whether `varying` represents grid sizes ("N") or omega values ("O") (default="N").

    Returns:
        dict: Maps each grid size (or omega value) to the mean and variance of iterations required.
        dict:  Maps each grid size (or omega value) to an example converged grid (with objects)
        list: Reference values from running SOR on an empty grid (no objects).
    """

    all_results = dict()
    all_results_map = dict()
    zeros_metric = []

    ntje = N
    omega = omegatje
    # iterate over all grid-sizes
    for variable in varying:
        # determine what value we're iterating over
        if what_value == "N":
            ntje = variable
        elif what_value == "O":
            omega = variable
        else:
            raise ValueError(f"{what_value} is not a valid variable to vary")
        if ntje < 20:
            continue
        print(
            f"starting parallel implementation of SOR for grid size {ntje}x{ntje}, omega: {omega}"
        )
        result_config = dict()
        all_solution_map = dict()
        # loop over different object grid configurations (object sizes)
        for config in range(len(all_grids[ntje])):
            # loop over the number of grids per grid-setting (parallel implementation)
            pars = []

            # make parameter list for parallelization
            for run in range(num_grids):
                pars.append((ntje, tol, max_iters, omega, all_grids[ntje][config][run]))

            # parallelizaiton
            with Pool(PROCESSES) as pool:
                assert PROCESSES < os.cpu_count(), (
                    f"Lower the number of processes {PROCESSES}"
                )
                all_sols = pool.map(non_sequential_SOR, pars)
                iter1, map2 = zip(*all_sols)
                itertjes = list(iter1)
                soltjes = list(map2)
                assert np.any(itertjes) < max_iters, (
                    f"maximum number of iterations for variables N:{ntje}, omega:{omega}, config{config} is reached, choose different omega"
                )

            # calculate mean and variance, save for every grid size and object configuration
            mean_config = np.mean(itertjes)
            var_config = np.var(itertjes)
            all_solution_map[object_configs[config]] = soltjes[0]
            result_config[object_configs[config]] = (mean_config, var_config)
        all_results_map[variable] = all_solution_map
        all_results[variable] = result_config

        # a null-measure: with no objects on the grid
        pars = (ntje, tol, max_iters, omega, None)
        zero_sol, _ = sequential_SOR(ntje, tol, max_iters, omega, None)
        zeros_metric.append(zero_sol)
    return all_results, all_results_map, zeros_metric


def statistical_test_for_objects(object_configs, all_res, forwhat="O"):
    """
    Performs statistical tests on object configurations and writes results to a file.

    This function analyzes different object configurations and evaluates their statistical
    properties based on the results obtained for various omega values.

    Parameters:
        object_configs (list): List of object configurations.
        all_res (dict): Dictionary containing results for different omega values.
        for_what: for "O" (varying omega experiments) or "N" (varying grid-size experiments)

    """

    # Ensure the "data" folder exists
    output_file = f"data/statistical_ttest_{forwhat}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        # Object configurations
        c_object012 = object_configs[:3]
        c_object23 = object_configs[2:4]
        if forwhat == "O":
            testing_what = "Omega"
        elif forwhat == "N":
            testing_what = "Grid Sizes"
        else:
            raise ValueError(
                f"Can't do statistical tests, experiment {forwhat} does not exist, choose O or N"
            )
        # Statistical testing for all omega values
        for value, configs in all_res.items():
            f.write(f"Statistical testing for {testing_what} = {value}\n")
            f.write("-" * 50 + "\n")

            # Extract means and variances
            data_012 = [configs[obj_config] for obj_config in c_object012]
            data_23 = [configs[obj_config] for obj_config in c_object23]

            # Test for different object configurations (same surface, different sizes)
            f.write("Different number of objects on the grid (same surface)\n")
            if len(data_012) == 3:
                mean0, var0 = data_012[0]
                mean1, var1 = data_012[1]
                mean2, var2 = data_012[2]

                t_stat_01, p_value_01 = ttest_ind_from_stats(
                    mean0, np.sqrt(var0), 10, mean1, np.sqrt(var1), 10, equal_var=False
                )
                f.write(
                    f"T-test for Object Configs {object_configs[0]} & {object_configs[1]}: t={t_stat_01:.3f}, p={p_value_01:.3f}\n"
                )

                t_stat_12, p_value_12 = ttest_ind_from_stats(
                    mean1, np.sqrt(var1), 10, mean2, np.sqrt(var2), 10, equal_var=False
                )
                f.write(
                    f"T-test for Object Configs {object_configs[1]} & {object_configs[2]}: t={t_stat_12:.3f}, p={p_value_12:.3f}\n"
                )

                t_stat_02, p_value_02 = ttest_ind_from_stats(
                    mean0, np.sqrt(var0), 10, mean2, np.sqrt(var2), 10, equal_var=False
                )
                f.write(
                    f"T-test for Object Configs {object_configs[0]} & {object_configs[2]}: t={t_stat_02:.3f}, p={p_value_02:.3f}\n"
                )

            # Test for different object configurations (different surface covered)
            f.write("\nDifferent surface covered\n")
            if len(data_23) == 2:
                mean2, var2 = data_23[0]
                mean3, var3 = data_23[1]

                t_stat_23, p_value_23 = ttest_ind_from_stats(
                    mean2, np.sqrt(var2), 10, mean3, np.sqrt(var3), 10, equal_var=False
                )
                f.write(
                    f"T-test for Object Configs {object_configs[2]} & {object_configs[3]}: t={t_stat_23:.3f}, p={p_value_23:.3f}\n"
                )

            f.write("=" * 50 + "\n")

    print(f"Statistical results saved to {output_file}")
