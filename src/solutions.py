import numpy as np 

def spat_approx_1a(deltax, time, solutions, x, L):
    return (solutions[2] - 2*solutions[1] + solutions[0])/np.power(deltax, 2)

def time_approx_1a(deltat, time, func, x):
    return (func(x, time+deltat) - 2*func(x, time) + func(x, time-deltat))/np.power(deltat, 2)

def one_b(which_one):

    def b_one(x):
        return np.sin(2*np.pi*x)

    def b_two(x):
        return np.sin(5*np.pi*x)

    def b_three(x):
        if x > 1/5 and x < 2/5:
            return np.sin(5*np.pi*x)
        else:
            return 0

    L = 1
    N = 100
    c=1
    time = 0

    deltax = L/N
    deltat = 0.001
    xs = np.arange(0, L, deltax)

    if which_one==1:
        func = b_one
    elif which_one==2:
        func=b_two
    elif which_one == 3: 
        func=b_three
    else:
        raise ValueError(f"invalid option {which_one}, choose option 1, 2 or 3 (integer form)")
    
    
    #saving the solutions
    sols_prev = [func(xje) for xje in xs]
    sols = sols_prev.copy()

    sols_next = np.zeros(len(xs))

    overall_solutions = []

    #the full function 
    for i in range(30000):
        for j,x in enumerate(xs):
            if j == 0: 
                sols_next[j]= 0
            elif j == len(xs) -1:
                sols_next[j] = 0
            else:
                sols_next[j] = np.power(deltat, 2) * np.power(c, 2) * spat_approx_1a(deltax, time, (sols[j-1], sols[j], sols[j+1]), x, L) + 2*sols[j] - sols_prev[j]
        if i%3000 == 0: 
            overall_solutions.append(sols)
        sols_prev = sols_next.copy()
        sols = sols_next.copy()
    return overall_solutions, xs