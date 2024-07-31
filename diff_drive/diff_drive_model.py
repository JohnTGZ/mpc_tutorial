from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def export_diff_drive_model() -> AcadosModel:
    """Differential wheel drive model

    Returns:
        AcadosModel: _description_
    """
    model_name = 'diff_drive'

    # Constants
    I = 1.0 # Axle Track
     = 1.0 # Wheel radius
    
    # States 

    #####
    # Set up states 
    #####
    x = SX.sym('x') # x position
    y = SX.sym('y') # y position
    theta = SX.sym('theta') # Yaw angle

    # theta = SX.sym('theta') # Angular position of pendulum 
    # v1 = SX.sym('v1') # Cart velocity
    # dtheta = SX.sym('dtheta') # Angular velocity of pendulum

    x = vertcat(x1, theta, v1, dtheta)  # System state

    #####
    # Set up controls 
    #####

    dphi_l = SX.sym('dphi_l') # Left wheel velocity
    dphi_r = SX.sym('dphi_r') # Right wheel velocity
    u = vertcat(dphi_l, dphi_r)  # Control input

    # Derivative of system state x: x_dot
    x1_dot = SX.sym('x1_dot')
    theta_dot = SX.sym('theta_dot')
    v1_dot = SX.sym('v1_dot')
    dtheta_dot = SX.sym('dtheta_dot')

    xdot = vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot) 

    #####
    # dynamics
    #####

    # f_expl: explicit forces
    f_expl = vertcat()

    # f_impl: implicit forces
    f_impl = xdot - f_expl

    # Define the model
    model = AcadosModel()
    
    model.f_impl_expr = f_impl # Implicit dynamics
    model.f_expl_expr = f_expl # explicit dynamics
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # Store meta information
    model.x_labels = ['$x$ [m]', r'$\theta$ [rad]', '$v$ [m]', r'$\dot{\theta}$ [rad/s]']
    model.u_labels = ['$F$']
    model.t_label = '$t$ [s]'
    
    return model
