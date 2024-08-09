from acados_template import AcadosSim, AcadosSimSolver
from diff_drive_model import export_diff_drive_model
from utils import plot_diff_drive, plot_diff_drive_trajectory
import numpy as np

sim = AcadosSim()
sim.model = export_diff_drive_model()

Tf = 0.1 # Time horizon
nx = sim.model.x.rows() # Number of states
nu = sim.model.u.rows() # Number of control inputs
N_sim = 300

# set simulation time
sim.solver_options.T = Tf
# set options
sim.solver_options.integrator_type = 'IRK'
sim.solver_options.num_stages = 3
sim.solver_options.num_steps = 3
sim.solver_options.newton_iter = 3 # for implicit integrator
sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

# create simulation solver
acados_integrator = AcadosSimSolver(sim)

# Initial state
x0 = np.array([0.0,     # x
                0.0,    # y
                0.0,    # theta
                0.0,    # phi1
                0.0     # phi2
                ]) 

# Initial control input
# u0 = np.zeros((nu,))             # Initial speed
u0 = np.array([1.5,     # tau_1
               -1.5      # tau_2
               ])                    

# xdot_init = np.zeros((nx,))             # Initial speed
xdot_init = np.array([  0.0,     # x_dot
                        1.0,    # y_dot
                        0.0,    # theta_dot
                        1.0,    # phi1_dot
                        0.0     # phi2_dot
                        ]) # Initial state             

simX = np.zeros((N_sim+1, nx))          
simU = np.zeros((N_sim+1, nu))     # Simulated input

simX[0,:] = x0
simU[0,:] = u0

for i in range(N_sim):
    # Note that xdot is only used if an IRK integrator is used
    simX[i+1,:] = acados_integrator.simulate(x=simX[i,:], u=u0, xdot=xdot_init)

# S_forw = acados_integrator.get("S_forw")
# print("S_forw, sensitivities of simulation result wrt x,u:\n", S_forw)

t = np.linspace(0, N_sim*Tf, N_sim+1) # Time array

# print(f"t: {t.shape}")
# print(f"np.repeat(u0, N_sim): {(np.repeat(u0, N_sim)).shape}")

# plot_diff_drive(t, 
#                 2,                             # u_max
#                 simU,                           # U
#                 simX,                           # X_true
#                 latexify=False, 
#                 time_label=sim.model.t_label, 
#                 x_labels=sim.model.x_labels, 
#                 u_labels=sim.model.u_labels)

plot_diff_drive_trajectory( t, 
                            2,                              # u_max
                            simU,                           # U
                            simX,                           # X_true
                            time_label=sim.model.t_label, 
                            x_labels=sim.model.x_labels, 
                            u_labels=sim.model.u_labels)

