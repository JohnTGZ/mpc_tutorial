from acados_template import AcadosSim, AcadosSimSolver
from bicycle_model import export_bicycle_model
from utils import plot_bicycle_model_states, bicycle_animation
import numpy as np

sim = AcadosSim()
sim.model = export_bicycle_model()

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
x0 = np.array([-2.0,    # s
                0.0,    # n
                0.0,    # alpha
                0.0,    # v
                0.0,    # D
                0.0])   # delta

# Initial control input
u0 = np.array([0.1,     # derD
               2.0      # derDelta
               ])                    

xdot_init = np.array([-2.0,     # sdot
                        0.0,    # ndot
                        0.0,    # alphadot
                        0.0,    # vdot
                        0.0,    # Ddot
                        0.0])   # deltadot

simX = np.zeros((N_sim+1, nx))          
simU = np.zeros((N_sim+1, nu))     # Simulated input

simX[0,:] = x0
simU[0,:] = u0

for i in range(N_sim):
    # Note that xdot is only used if an IRK integrator is used
    simX[i+1,:] = acados_integrator.simulate(x=simX[i,:], u=u0, xdot=xdot_init)

S_forw = acados_integrator.get("S_forw")
print("S_forw, sensitivities of simulation result wrt x,u:\n", S_forw)

t = np.linspace(0, N_sim*Tf, N_sim+1) # Time array

# print(f"t: {t.shape}")
# print(f"np.repeat(u0, N_sim): {(np.repeat(u0, N_sim)).shape}")

# plot_bicycle_model_states(t, 
#                             2,                             # u_max
#                             simU,                           # U
#                             simX,                           # X_true
#                             latexify=False, 
#                             time_label=sim.model.t_label, 
#                             x_labels=sim.model.x_labels, 
#                             u_labels=sim.model.u_labels)

# Animate car
bicycle_animation(t, simX)

