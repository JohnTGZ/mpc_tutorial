from acados_template import AcadosOcp, AcadosOcpSolver
from diff_drive_model import export_diff_drive_model
import numpy as np
import casadi as ca
from utils import plot_diff_drive, plot_diff_drive_trajectory

ocp = AcadosOcp()

model = export_diff_drive_model()
ocp.model = model

Tf = 2.0 # Prediction horizon
nx = model.x.rows() # Number of state variables
nu = model.u.rows() # Number of control variables
N = 10

# Set number of shooting intervals
ocp.dims.N = N
# Set prediction horizon
ocp.solver_options.tf = Tf

# Cost matrices
# 
Q_mat = 2 * np.diag([1e3, 1e3, 1e3, 1e3, 1e3])   # weights on state 
R_mat = 2 * np.diag([1e-2, 1e-2])                   # weights on controls

# Path cost
ocp.cost.cost_type = "NONLINEAR_LS" # EXTERNAL, LINEAR_LS, NONLINEAR_LS, CONVEX_OVER_NONLINEAR
ocp.model.cost_y_expr = ca.vertcat(model.x, model.u) #CasADi expression for nonlinear least squares
ocp.cost.yref = np.zeros((nx+nu,)) #  reference at intermediate shooting nodes (1 to N-1).
ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

# Terminal cost
ocp.cost.cost_type_e = "NONLINEAR_LS"
ocp.cost.yref_e = np.zeros((nx,)) # cost reference at terminal shooting node (N).
ocp.model.cost_y_expr_e = model.x
ocp.cost.W_e = Q_mat

# Set constraints
Fmax = 1000
ocp.constraints.lbu = np.array([-Fmax, -Fmax]) # Lower bound on controls
ocp.constraints.ubu = np.array([+Fmax, +Fmax]) # Upper bound on controls
ocp.constraints.idxbu = np.array([0,1]) #Indices of bounds on u (defines Jbu) at shooting nodes (0 to N-1). Can be set by using Jbu

# Initial state constraint
ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# set options
ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
# PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
# PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
ocp.solver_options.integrator_type = 'IRK'
# ocp.solver_options.print_level = 1
ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

ocp_solver = AcadosOcpSolver(ocp)

simX = np.zeros((N+1, nx))
simU = np.zeros((N+1, nu))

# Set reference trajectory y_ref of type np.ndarray, (n_hrzn+1, nx + nu) 
for i in range(N): # For each step
  self.acados_ocp_solver.cost_set(i, 'yref', y_ref[i, :])

status = ocp_solver.solve()
ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")
# iter: iteration number
# res_stat: stationarity residual
# res_eq: residual wrt equality constraints (dynamics)
# res_ineq: residual wrt inequality constraints (constraints)
# res_comp: residual wrt complementarity conditions
# qp_stat: status of QP solver
# qp_iter: number of QP iterations
# alpha: SQP step size
# qp_res_stat: stationarity residual of the last QP solution
# qp_res_eq: residual wrt equality constraints (dynamics) of the last QP solution
# qp_res_ineq: residual wrt inequality constraints (constraints) of the last QP solution
# qp_res_comp: residual wrt complementarity conditions of the last QP solution

if status != 0:
    raise Exception(f'acados returned status {status}.')
    # 0 – success
    # 1 – failure
    # 2 – maximum number of iterations reached
    # 3 – minimum step size in QP solver reached
    # 4 – qp solver failed

# Get solution
for i in range(N): # For each step
    simX[i,:] = ocp_solver.get(i, "x")
    simU[i,:] = ocp_solver.get(i, "u")
simX[N,:] = ocp_solver.get(N, "x")

print(f"simX[0,:]: {simX[0,:]}")
print(f"simX[N,:]: {simX[N,:]}")

plot_diff_drive(np.linspace(0, Tf, N+1), Fmax, simU, simX, latexify=True, time_label=model.t_label, x_labels=model.x_labels, u_labels=model.u_labels)

t = np.linspace(0, N_sim*Tf, N_sim+1) # Time array

plot_diff_drive_trajectory( t, 
                            2,                              # u_max
                            simU,                           # U
                            simX,                           # X_true
                            time_label=sim.model.t_label, 
                            x_labels=sim.model.x_labels, 
                            u_labels=sim.model.u_labels)
