from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def export_diff_drive_model() -> AcadosModel:
    """Differential wheel drive model

    Returns:
        AcadosModel: _description_
    """
    model_name = 'diff_drive'

    # Constants
    W = 0.16/2 # Axle Track
    rho = 0.066 # Wheel radius
    
    # States 

    #####
    # Set up states 
    #####
    x = SX.sym('x') # x position
    y = SX.sym('y') # y position
    theta = SX.sym('theta') # Yaw angle
    phi1 = SX.sym('phi1') # Right wheel angular position
    phi2 = SX.sym('phi2') # Left wheel angular position

    x_state = vertcat(x, y, theta, phi1, phi2)  # System state

    x_dot = SX.sym('x_dot') # x velocity
    y_dot = SX.sym('y_dot') # y velocity
    theta_dot = SX.sym('theta_dot') # time derivative of yaw
    phi1_dot = SX.sym('phi1_dot') # Right wheel angular velocity
    phi2_dot = SX.sym('phi2_dot') # Left wheel angular velocity

    x_state_dot = vertcat(x_dot, y_dot, theta_dot, phi1_dot, phi2_dot)  # Time derivative of system state

    #####
    # Set up controls 
    #####

    tau1 = SX.sym('tau1') # Right wheel torque
    tau2 = SX.sym('tau2') # Left wheel torque
    u = vertcat(tau1, tau2)  # Control input

    #####
    # dynamics
    #####
    # https://github.com/acados/acados/wiki/How-to:-Dynamic-System-Models-and-Integrators-in-acados

    # f_expl: explicit forces in the form f_expl(x, u) = xdot 
    #   used by the sim_erk_integrator
    f_expl = vertcat(   (rho/2) * cos(theta) * (phi1_dot - phi2_dot),      # x_dot
                        (rho/2) * sin(theta) * (phi1_dot - phi2_dot),      # y_dot
                        (rho/2) * -(phi1_dot + phi2_dot)/W,                 # theta_dot
                        tau1,                                                # phi1_dot
                        tau2)                                               # phi2_dot

    # f_impl: implicit forces in the form f_impl(x, xdot, u, [z]) = 0
    #   these type of formulation is used by the implicit integrators: sim_irk_integrator, sim_lifted_irk_integrator, sim_new_lifted_irk_integrator
    #   z: algebraic state variables (an optional part of the model, currently only supported by sim_irk_integrator)
    f_impl = x_state_dot - f_expl

    # Define the model
    model = AcadosModel()
    
    model.f_impl_expr = f_impl # Implicit dynamics
    model.f_expl_expr = f_expl # explicit dynamics
    model.x = x_state             # System state
    model.xdot = x_state_dot       # Time derivative of system state
    model.u = u             # Control input
    model.name = model_name

    # Store meta information
    model.x_labels = ['$x$ [m]', '$y$ [m]', r'$\theta$ [rad]', r'$\phi_1$ [rad]', r'$\phi_2$ [rad]',]
    model.u_labels = [r'$\tau_1$ [N]', r'$\tau_2$ [N]']
    model.t_label = '$t$ [s]'
    
    return model
