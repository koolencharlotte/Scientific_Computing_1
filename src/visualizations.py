import matplotlib.pyplot as plt

def visualization_1b(overall_solutions, xs):
    plt.figure(figsize=(4,4))
    for k in range(len(overall_solutions)):
        plt.plot(xs, overall_solutions[k])

    plt.title("Wave Time-stepping Approximation")
    plt.xlabel("L")
    plt.ylabel(r'$\Psi^n$')
    plt.show()