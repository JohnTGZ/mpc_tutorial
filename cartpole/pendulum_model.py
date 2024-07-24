#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def export_pendulum_ode_model() -> AcadosModel:
    """Cartpole model

    Returns:
        AcadosModel: _description_
    """
    model_name = 'pendulum_ode'

    # Constants
    M = 1.0 # Mass of cart [kg]
    m = 0.1 # Mass of the ball [kg]
    g = 9.81 # Gravitational constant [m/s^2]
    l = 0.8 # Length of the rod [m]

    #####
    # Set up states & controls
    #####
    x1 = SX.sym('x1') # Cart position
    theta = SX.sym('theta') # Angular position of pendulum 
    v1 = SX.sym('v1') # Cart velocity
    dtheta = SX.sym('dtheta') # Angular velocity of pendulum

    x = vertcat(x1, theta, v1, dtheta)  # System state

    F = SX.sym('F') # Force
    u = vertcat(F)  # Control input

    # Derivative of system state x: x_dot
    x1_dot = SX.sym('x1_dot')
    theta_dot = SX.sym('theta_dot')
    v1_dot = SX.sym('v1_dot')
    dtheta_dot = SX.sym('dtheta_dot')

    xdot = vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot) 

    #####
    # dynamics
    #####
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    denominator = M + m - m*cos_theta*cos_theta

    # f_expl: explicit forces
    f_expl = vertcat(v1,
                     dtheta,
                     (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(M+m)*g*sin_theta)/(l*denominator)
                     )

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
