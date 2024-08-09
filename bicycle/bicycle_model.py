import numpy as np
from acados_template import AcadosModel
from casadi import MX, vertcat, sin, cos, tanh, interpolant

from tracks.readDataFcn import getTrack

def export_bicycle_model() -> AcadosModel:
    """Differential wheel drive model

    Returns:
        AcadosModel: _description_
    """
    model_name = 'bicycle_model'

    # Load track parameters
    # s0: arc length of track
    # kapparef: curvature of track
    track = "LMS_Track.txt"
    [s0, xref, yref, psiref, kapparef] = getTrack(track)
    num_pts = len(s0) # Length of track
    pathlength = s0[-1]
    # Copy loop to beginning and end
    s0 = np.append(s0, [s0[num_pts-1] + s0[1:num_pts]])
    kapparef = np.append(kapparef, kapparef[1:num_pts])
    s0 = np.append([-s0[num_pts - 2] + s0[num_pts - 81 : num_pts - 2]], s0)
    kapparef = np.append(kapparef[num_pts - 80 : num_pts - 1], kapparef)

    # Creates lookup table for spline interpolations, example usage: kapparef_s(value)
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)

    ## Race car parameters
    m = 0.043       
    C1 = 0.5        
    C2 = 15.5       
    Cm1 = 0.28      
    Cm2 = 0.05      
    Cr0 = 0.011     
    Cr2 = 0.006     
    Cr3 = 5.0    
    Lr = 0.5 # Length from rear wheel to center of gravity
    Lf = 0.5 # Length from center of gravity to front wheel

    #####
    # Set up states 
    #####
    s = MX.sym("s")             # s: Arc length of track
    n = MX.sym("n")             # n: Minimal distance between vehicle and the center line
    alpha = MX.sym("alpha")     # alpha: Orientation of vehicle relative to track
    v = MX.sym("v")             # v: Velocity
    D = MX.sym("D")             # D: Duty cycle
    delta = MX.sym("delta")     # delta Steering angle of vehicle
    x = vertcat(s, n, alpha, v, D, delta)

    # xdot
    sdot = MX.sym("sdot")
    ndot = MX.sym("ndot")
    alphadot = MX.sym("alphadot")
    vdot = MX.sym("vdot")
    Ddot = MX.sym("Ddot")
    deltadot = MX.sym("deltadot")
    xdot = vertcat(sdot, ndot, alphadot, vdot, Ddot, deltadot)

    #####
    # Set up controls 
    #####

    derD = MX.sym("derD")           # Time derivative of duty cycle
    derDelta = MX.sym("derDelta")   # Time derivative of steering angle
    u = vertcat(derD, 
                derDelta)

    # algebraic variables
    z = vertcat([])

    # parameters
    p = vertcat([])

    #####
    # dynamics
    #####
    # https://github.com/acados/acados/wiki/How-to:-Dynamic-System-Models-and-Integrators-in-acados

    # f_expl: explicit forces in the form f_expl(x, u) = xdot 
    #   used by the sim_erk_integrator
    
    # m = 0.043       
    # C1 = 0.5        
    # C2 = 15.5       
    # Cm1 = 0.28      
    # Cm2 = 0.05      
    # Cr0 = 0.011     
    # Cr2 = 0.006     
    # Cr3 = 5.0    

    # F_d_x: Longitudinal force
    F_d_x = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(Cr3 * v)
    # beta: side-slip angle
    beta = (Lr  / (Lr + Lf)) * delta 

    sdota = v * cos(alpha + beta) / (1 - kapparef_s(s) * n)

    f_expl = vertcat(   sdota,                                              # sdot
                        v * sin(alpha + beta),                              # ndot
                        (v / Lr) * sin(beta) - kapparef_s(s) * sdota,         # alphadot
                        (F_d_x / m) * cos(beta),                            # vdot
                        derD,                                               # Ddot
                        derDelta)                                           # deltadot

    # f_impl: implicit forces in the form f_impl(x, xdot, u, [z]) = 0
    #   these type of formulation is used by the implicit integrators: sim_irk_integrator, sim_lifted_irk_integrator, sim_new_lifted_irk_integrator
    #   z: algebraic state variables (an optional part of the model, currently only supported by sim_irk_integrator)
    f_impl = xdot - f_expl


    #####
    # constraint on forces
    #####
    # Assume no tire slip
    a_lat = v * v * sin(beta) + F_d_x * sin(beta) / m   # Lateral force
    a_long = F_d_x / m                                  # Longitudinal force

    # Define the model
    model = AcadosModel()
    
    model.f_impl_expr = f_impl # Implicit dynamics
    model.f_expl_expr = f_expl # explicit dynamics
    model.x = x             # System state
    model.xdot = xdot       # Time derivative of system state
    model.u = u             # Control input
    model.z = z             # Algebraic variables
    model.p = p             # Parameters
    model.name = model_name

    # Store meta information
    model.x_labels = ['$s$ [m]', '$n$ [m]', r'$\alpha$ [rad]', r'$v$ [m/s]', r'$D$ [\%]', r'$\delta$ [rad]',]
    model.u_labels = [r'$\dot{D}$ [\%/s]', r'$\dot{\delta}$ [rad/s]']
    model.t_label = '$t$ [s]'

    # Constraints

    # con_h_expr: Nonlinear constraints on the initial shooting node
    # lh <= h(x, u) <= uh
    model.con_h_expr = vertcat(
        a_long,
        a_lat, 
        n,
        D,          
        delta
    )
    
    return model
