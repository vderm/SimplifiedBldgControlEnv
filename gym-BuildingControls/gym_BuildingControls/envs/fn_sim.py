#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Vasken Dermardiros

Essential functions to run simulations, à la Vasken
"""

# Import dependencies
# import numpy as np
from numpy import array, reshape, multiply, dot, cos, pi, zeros, shape, tile, clip
from numpy import random
# from numpy.linalg import solve

# Calculate temperatures for next timesteps: T(t+1) = U^-1 * Q(t)
# Seperated in case the explicit scheme is needed
def calcT(U_inv, F, C, Qint, Tt, TKt, dt):
    # Q-vector: Q = Qin + F*TM(t) + C/dt*T(t)
    nN = U_inv.shape[0]
    Q = Qint + reshape(dot(F,TKt),(nN,1)) + multiply(C/dt,reshape(Tt,(nN,1)))
    return dot(U_inv, Q).T

# Function to calculate future temperatures
def futureT(Q, initialT, TK, U_inv, F, C, nN, dt):
    ft, nM = shape(TK)           # Temperature matrices
    T = zeros((ft, nN)) # degC
    T[0,] = initialT
    for i in range(ft-1):  # Calculate future states
        T[i+1,] = calcT(U_inv, F, C, Q[i,].reshape(nN,1), T[i,], TK[i+1,].reshape(nM,1), dt)
    return T

# Generate a periodic input
def periodic(mean, peak_to_peak, peak_time, period, dt, nt, days=1):
    # peak_time in hour of day; 3PM is 15
    # {period, dt, nt} in seconds; 1 day = 86400s
    # days in days, obviously
    theta = peak_time*pi/12
    omega = 2*pi/period
    return array(days*[mean + peak_to_peak/2*cos(omega*dt*t - theta) for t in range(nt)])

# Generate a periodic input
def halfperiodic(peak, peak_time, period, dt, nt, days=1):
    # peak_time in hour of day; 3PM is 15
    # {period, dt, nt} in seconds; 1 day = 86400s
    # days in days, obviously
    theta = peak_time*pi/12
    omega = 2*pi/period
    return array(clip(days*[peak*cos(omega*dt*t - theta) for t in range(nt)], 0, None))

# Linear ramps
def linearRamp(T_SP_day, T_SP_dT, setback_beg, setback_end, ramp_dur, dt, nt, days=1):
    # T_SP_day:    daytime setpoint to maintain
    # T_SP_dT:     setback amount
    # setback_beg: setback beginning time (occupancy departure)
    # setback_end: setback end time (occupancy arrival)
    # ramp_dur:    hours, ramp duration; "0" equals to a step change
    T_SP = zeros((nt,1)) # degC; Interior temperature setpoint per timestep
    for t in range(nt):
        time = t*dt/3600.
        if (setback_beg <= time and time < (setback_beg+ramp_dur)):             # begin setback
            T_SP[t] = T_SP_day - (time-setback_beg)*T_SP_dT/ramp_dur
        elif ((setback_beg+ramp_dur) <= time or time < (setback_end-ramp_dur)): # night time
            T_SP[t] = T_SP_day - T_SP_dT
        elif ((setback_end-ramp_dur) <= time and time < setback_end):           # revert setback
            T_SP[t] = T_SP_day - (setback_end-time)*T_SP_dT/ramp_dur
        else:                                                                   # day time
            T_SP[t] = T_SP_day
    return tile(T_SP, (days,1))

def random_TK(dt, nt):
    # Using reasonable pre-set parameters
    return periodic(random.uniform(-5,20), random.poisson(2.5), random.randint(0,23), 86400., dt, nt)

def random_TK_and_Q(dt, nt, TKpad=0):
    # Using reasonable pre-set parameters
    peak_time = random.randint(0,23)
    peak_time_solar = peak_time - 3 # typically 3 hours before peak temperature
    TK = periodic(random.uniform(-5,25), random.poisson(2.5), peak_time, 86400., dt, nt+TKpad)
    Q  = halfperiodic(random.uniform(0,2200), peak_time_solar, 86400., dt, nt)
    return TK, Q
