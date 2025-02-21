
# **Scientific Computing: Wave and Diffusion Simulations**
By Bart Koedijk (15756785) & Charlotte Koolen (15888592) & Chris Hoynck van Papendrecht (15340791)

This repository contains a set of numerical experiments and implementations for solving **wave** and **diffusion equations** using **finite difference methods** and **iterative solvers**.

## **Project Overview**
The project explores:
- The **wave equation** and its discretization using **finite difference methods**.
- The **diffusion equation**, including **time-dependent and steady-state solutions**.
- **Iterative solvers** such as **Jacobi, Gauss-Seidel, and Successive Over-Relaxation (SOR)** for solving the steady-state diffusion equation.
- The effect of **different object configurations** on convergence behavior.

## **Main Components**
- **`main.ipynb`** – A Jupyter Notebook that guides through all equations, numerical methods, and experimentation.
- **`src/solutions.py`** – Implementations of the wave and diffusion solvers.
- **`src/visualizations.py`** – Functions for plotting simulation results.
- **`data/`** – Directory for storing generated simulation data.
- **`plots/`** – Directory for saving visualizations.

## **Installation & Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/koolencharlotte/Scientific_Computing_1.git
   cd Scientific_Computing_1
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate   # On Windows use: myenv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**
Run the Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```
Follow the step-by-step explanations and visualizations inside the notebook.

