#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: Vasken Dermardiros

Variable Names
Uin: Conductance matrix input by user, upper triangle only, (nN x nN) (W/K)
U: Conductance matrix (symmetrical) with added capacitance for diagonal term, (nN x nN) (W/K)
U_inv: inverse of U; U^-1
C: Capacitance vector, (nN x 1) (J/K)
F: Conductance matrix of nodes connected to a known temperature source, (nN x nM) (W/K)
nN: Number of nodes
nM: Number of nodes with known temperatures / boundaries

These models are made for convenience. The method is generalized to any number
of interior and boundary nodes as in 'mUxFxCx' model definition below.
"""

def mF1C1(F_in, C_in, dt):
    """Simple room modeled as a single internal node with capacitance. Connects to
    exterior (F-matrix) with an effective conductance representing the whole
    wall with windows with air infiltration.

    Node Number: Object
    0: effective room node, connected to capacitor and T_ambient (in the Ms)

    Node Number with known temperatures: Object
    0: ambient air
    """
    # Load dependencies
    from numpy import zeros
    from numpy import sum as npsum
    from numpy.linalg import inv

    # #### Control
    nN = 1          # number of nodes
    nM = 1          # number of nodes with known temperatures

    #%% Nodal Connections
    # Declare variables
    Uin = zeros((nN,nN))     # W/K
    F = zeros((nN,nM))       # W/K
    C = zeros((nN,1))        # J/K

    # How are the nodes connected?
    # Uin[0,1] = (1/R + U + dx/kA)**-1

    # Connected to temperature sources
    F[0,0] = F_in

    # Nodes with capacitance
    C[0] = C_in

    #%% U-matrix completion, and its inverse
    U = -Uin - Uin.T  # U is symmetrical, non-diagonals are -ve
    s = -npsum(U,1)
    for i in range(0,nN):
        U[i,i] = s[i] + npsum(F[i,]) + C[i]/dt
    Uinv = inv(U)

    #%% Ship it
    return (Uinv, F, C, nN, nM)


def mU1F1C2(U_in, F_in, C_in, C_slab, dt):
    """ Model of a simple room that has heating/cooling applied in a different node
    than that of the air, eg.: a radiant slab system.

    Node Number: Object
    0: room air node, connected to ambient air (F0) node
    1: under slab node, connected to capacitor 1 (slab) and Node 0

    Node Number with known temperatures: Object
    0: ambient air

    External input:
    U_in: conductance under slab to slab surface
    F_in: conductance room air to slab surface
    C_in: capacitance of air
    C_slab: capacitance of slab
    """
    # Load dependencies
    from numpy import zeros
    from numpy import sum as npsum
    from numpy.linalg import inv

    nN = 2          # number of nodes
    nM = 1          # number of nodes with known temperatures

    #%% Nodal Connections
    # Declare variables
    Uin = zeros((nN,nN))     # W/K
    F = zeros((nN,nM))       # W/K
    C = zeros((nN,1))        # J/K

    # How are the nodes connected?
    Uin[0,1] = U_in

    # Connected to temperature sources
    F[0,0] = F_in

    # Nodes with capacitance
    C[0] = C_in
    C[1] = C_slab

    #%% U-matrix completion, and its inverse
    U = -Uin - Uin.T  # U is symmetrical, non-diagonals are -ve
    s = -npsum(U,1)
    for i in range(0,nN):
        U[i,i] = s[i] + npsum(F[i,]) + C[i]/dt
    Uinv = inv(U)

    #%% Ship it
    return (Uinv, F, C, nN, nM)


def mUxFxCx(Uin, F, C, dt):
    """ Generic model.
    """
    # Load dependencies
    from numpy import zeros
    from numpy import sum as npsum
    from numpy.linalg import inv

    nN = len(Uin)          # number of nodes

    #%% U-matrix completion, and its inverse
    U = -Uin - Uin.T  # U is symmetrical, non-diagonals are -ve
    s = -npsum(U,1)
    for i in range(0,nN):
        U[i,i] = s[i] + npsum(F[i,]) + C[i]/dt
    Uinv = inv(U)

    #%% Ship it
    return (Uinv, F, C, nN, nM)
