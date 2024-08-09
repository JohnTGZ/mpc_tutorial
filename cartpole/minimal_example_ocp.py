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

from acados_template import AcadosOcp, AcadosOcpSolver
from pendulum_model import export_pendulum_ode_model
import numpy as np
import casadi as ca
from utils import plot_pendulum

def main():

    ocp = AcadosOcp()

    model = export_pendulum_ode_model()
    ocp.model = model

    Tf = 1.0 # Prediction horizon
    nx = model.x.rows() # Number of state variables
    nu = model.u.rows() # Number of control variables
    N = 20

    # Set number of shooting intervals
    ocp.dims.N = N
    # Set prediction horizon
    ocp.solver_options.tf = Tf

    # Cost matrices
    Q_mat = 2 * np.diag([1e3, 1e3, 1e-2, 1e-2])
    R_mat = 2 * np.diag([1e-2])

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
    Fmax = 80
    ocp.constraints.lbu = np.array([-Fmax]) # Lower bound on controls
    ocp.constraints.ubu = np.array([+Fmax]) # Upper bound on controls
    ocp.constraints.idxbu = np.array([0]) #Indices of bounds on u (defines Jbu) at shooting nodes (0 to N-1). Can be set by using Jbu

    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

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
    simU = np.zeros((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # Get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")

    print(f"simX[0,:]: {simX[0,:]}")
    print(f"simX[N,:]: {simX[N,:]}")

    plot_pendulum(np.linspace(0, Tf, N+1), Fmax, simU, simX, latexify=True, time_label=model.t_label, x_labels=model.x_labels, u_labels=model.u_labels)


if __name__ == '__main__':
    main()
