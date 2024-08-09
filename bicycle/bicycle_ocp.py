from acados_template import AcadosOcp, AcadosOcpSolver
from bicycle_model import export_bicycle_model
import numpy as np
import casadi as ca
from utils import bicycle_animation
from tracks.readDataFcn import getTrack

#####
# Set up model and track
#####

track = "LMS_Track.txt"
[Sref, _, _, _, _] = getTrack(track)

ocp = AcadosOcp()
ocp.model = export_bicycle_model()

#####
# Parameters
#####
Tf = 2.0            # Prediction horizon [s]
N = 10              # Number of discretization steps
T = 10.0            # Maximum simulation time [s]
sref_N = 3          # reference for final reference progress

#####
# Set dimensions
#####
nx = ocp.model.x.rows() # Number of state variables
nu = ocp.model.u.rows() # Number of control variables
ny = nx + nu      
ny_e = nx           # Terminal
Nsim = int(T * N / Tf)  # Number of simulations

# Set number of shooting intervals
ocp.dims.N = N

ns = 2
nsh = 2 # Number of slack variables

#####
# Cost matrices
#####
# Q: weights on path state 
Q = np.diag([ 1e-1, # s: Arc length of track
              1e-8,  # n: Minimal distance between vehicle and the center line
              1e-8, # alpha: Orientation of vehicle relative to track
              1e-8, # v: Velocity
              1e-3, # D: Duty cycle
              5e-3 ])   # delta Steering angle of vehicle

# R: weights on path controls 
R = np.eye(nu)
R[0, 0] = 1e-3
R[1, 1] = 5e-3

# Q: weights on terminal state 
Qe = np.diag([ 5e0, 
                1e1, 
                1e-8, 
                1e-8, 
                5e-3, 
                2e-3 ])

unscale = N / Tf

###
# Path cost
###
ocp.cost.cost_type = "LINEAR_LS" # EXTERNAL, LINEAR_LS, NONLINEAR_LS, CONVEX_OVER_NONLINEAR
ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)  # Weights
# state cost coefficients
Vx = np.zeros((ny, nx)) 
Vx[:nx, :nx] = np.eye(nx) # Set the upper left block matrix to be identity
ocp.cost.Vx = Vx  # Set state cost coefficients
# control cost coefficients
Vu = np.zeros((ny, nu))
Vu[6, 0] = 1.0
Vu[7, 1] = 1.0
ocp.cost.Vu = Vu
# ocp.model.cost_y_expr = ca.vertcat(model.x, model.u) #CasADi expression for nonlinear least squares

# DEfine coefficient of slack variables
ocp.cost.zl = 100 * np.ones((ns,))
ocp.cost.Zl = 0 * np.ones((ns,))
ocp.cost.zu = 100 * np.ones((ns,))
ocp.cost.Zu = 0 * np.ones((ns,))

###
# Terminal cost
###
ocp.cost.cost_type_e = "LINEAR_LS"
ocp.cost.W_e = Qe / unscale   # Weights
# state cost coefficients
Vx_e = np.zeros((ny_e, nx))
Vx_e[:nx, :nx] = np.eye(nx)
ocp.cost.Vx_e = Vx_e
# ocp.model.cost_y_expr_e = model.x  #CasADi expression for nonlinear least squares

# set intial references
ocp.cost.yref = np.array([1, 0, 0, 0, 0, 0, 0, 0]) # yref: Initial reference at intermediate shooting nodes (1 to N-1).
ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0]) # yref_e: Reference at terminal shooting node (N).

#####
# set constraints
#####
# Initial state constraint
ocp.constraints.x0 = np.array([-2, 0, 0, 0, 0, 0])

# Constraints on n
ocp.constraints.lbx = np.array([-12])
ocp.constraints.ubx = np.array([12])
ocp.constraints.idxbx = np.array([1])
# Constraints on control (throttle and steering angle)
ocp.constraints.lbu = np.array([model.dthrottle_min, model.ddelta_min])
ocp.constraints.ubu = np.array([model.dthrottle_max, model.ddelta_max])
ocp.constraints.idxbu = np.array([0, 1])
# Constraints on path
ocp.constraints.lh = np.array(
    [
      -4.0,     #a_long  
      -4.0,     #a_lat
      -0.12,    #n
      -1.0,    #D
      -2.0,     #delta
    ]
)
ocp.constraints.uh = np.array(
    [
      4.0,     #a_long  
      4.0,     #a_lat
      0.12,    #n
      1.0,    #D
      2.0,     #delta
    ]
)
# Constraints on 
# lsh: Lower bounds on slacks corresponding to soft lower bounds for nonlinear constraints
ocp.constraints.lsh = np.zeros(nsh)
ocp.constraints.ush = np.zeros(nsh)
ocp.constraints.idxsh = np.array([0, 2])

#####
# set solver options
#####
# set QP solver and integration
ocp.solver_options.tf = Tf
# ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM" # FULL_CONDENSING_QPOASES
# PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
# PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
ocp.solver_options.nlp_solver_type = "SQP_RTI"
ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.sim_method_num_stages = 4
ocp.solver_options.sim_method_num_steps = 3

# create solver
acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

#####
# initialize data structs
#####
simX = np.zeros((Nsim, nx))
simU = np.zeros((Nsim, nu))
s0 = ocp.model.x0[0]
tcomp_sum = 0
tcomp_max = 0

#####
# Simulation
#####
for i in range(Nsim):
  # Update references


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
