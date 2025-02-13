import numpy as np 

def spat_approx_1a(deltax, solutions):
    """
    Computes the second-order spatial approximation for the wave equation.

    Parameters:
    deltax (float): Spatial step size.
    solutions (tuple): Three consecutive solution values (previous, current, next).

    Returns:
    float: Second-order finite difference approximation.
    """

    assert deltax != 0, "can't divide by 0 (spatial approximation)"
    return (solutions[2] - 2*solutions[1] + solutions[0]) / np.power(deltax, 2)

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
        if x > 1/5 and x < 2/5:
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
        raise ValueError(f"Invalid option {which_one}, choose 1, 2, or 3")

    # Saving the solutions
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
            sols_next[j] = np.power(deltat, 2) * np.power(c, 2) * spat_approx_1a(deltax, (sols[j-1], sols[j], sols[j+1])) + 2 * sols[j] - sols_prev[j]

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

# sequential jacobi iteration
def sequential_jacobi(N, tol, max_iters):
    """
    Solves the Jacobi iteration using the update equation:
        c_{i,j}^{k+1} = (1/4) * (c_{i+1,j}^{k} + c_{i-1,j}^{k} + c_{i,j+1}^{k} + c_{i,j-1}^{k})
    """

    # grid initialisation
    c_old = np.zeros((N, N)) # N is max
    c_next = np.copy(c_old)

    # boundary conditions
    c0 = 0.0
    cL = 1.0

    # top boundary (y=1, j = N - 1)
    c_old[0, :] = cL  
    c_next[0, :] = cL
    
    # bottom boundary (y=0, j = 0)
    c_old[-1, :] = c0
    c_next[-1, :] = c0

    iter = 0
    delta = float('inf')

    while delta > tol and iter < max_iters:
        delta = 0

        for i in range(1, N-1):  # periodic in x
            for j in range(1, N-1):  # fixed in y
                
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
                south = c0 if j == 0 else c_old[i, j - 1]
                north = cL if j == N - 1 else c_old[i, j + 1]

                # Jacobi update equation
                c_next[i, j] = 0.25 * (west + east + south + north)

                delta = max(delta, abs(c_next[i, j] - c_old[i, j]))

        # swap matrices for next iter
        c_old[:], c_next[:] = c_next, c_old

        iter += 1

    print(f"Converged in {iter} iterations with δ = {delta:.8f}")

    return c_old

# sequential gauss seidel
def sequential_gauss_seidel(N, tol, max_iters):
    """
    Solves the Gauss-Seidel iteration using the update equation:
        c_{i,j}^{n+1} = (1/4) * (c_{i+1,j}^{k} + c_{i-1,j}^{k+1} + c_{i,j+1}^{k} + c_{i,j-1}^{k+1})
    """

    # grid initialisation
    c = np.zeros((N, N)) # N is max

    # boundary conditions
    c0 = 0.0
    cL = 1.0

    # top boundary (y=1, j = N - 1)
    c[0, :] = cL  
    
    # bottom boundary (y=0, j = 0)
    c[-1, :] = c0

    iter = 0
    delta = float('inf')

    while delta > tol and iter < max_iters:
        delta = 0

        for i in range(1, N-1):  # periodic in x
            for j in range(1, N-1):  # fixed in y

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

    print(f"Converged in {iter} iterations with δ = {delta:.8f}")

    return c





