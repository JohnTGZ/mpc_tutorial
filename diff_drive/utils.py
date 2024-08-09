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

import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot


def plot_diff_drive_trajectory(t, u_max, U, X_true, plt_show=True, time_label='$t$', x_labels=None, u_labels=None):
    """
    Params:
        t: time values of the discretization
        u_max: maximum absolute value of u for visualization
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
    """

    # nx = X_true.shape[1]
    # nu = U.shape[1]
    fig, ax = plt.subplots()
    ax.scatter(X_true[:, 0], X_true[:, 1], alpha=0.5)
    # axes[0].grid()
    ax.set_xlabel(f'$x$')
    ax.set_ylabel(f'$y$')

    plt.show()

def plot_diff_drive(t, u_max, U, X_true, latexify=False, plt_show=True, time_label='$t$', x_labels=None, u_labels=None):
    """
    Params:
        t: time values of the discretization
        u_max: maximum absolute value of u for visualization
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        latexify: latex style plots
    """
    if latexify:
        latexify_plot()

    nx = X_true.shape[1]
    nu = U.shape[1]
    fig, axes = plt.subplots(nx+nu, 1, sharex=True)

    for i in range(nx): # For each state
        axes[i].plot(t, X_true[:, i])
        axes[i].grid()
        if x_labels is not None:
            axes[i].set_ylabel(x_labels[i])
        else:
            axes[i].set_ylabel(f'$x_{i}$')

    # Make step plot in Last axes
    for i in range(nu):
        axes[nx+i].step(t, U[:, i])

        if u_labels is not None:
            axes[nx+i].set_ylabel(u_labels[i])
        else:
            axes[nx+i].set_ylabel(f'$u_{i}$')

        axes[nx+i].hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
        axes[nx+i].hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
        axes[nx+i].set_ylim([-1.2*u_max, 1.2*u_max])
        axes[nx+i].set_xlim(t[0], t[-1])
        axes[nx+i].grid()

    axes[-1].set_xlabel(time_label)


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    fig.align_ylabels()

    if plt_show:
        plt.show()
