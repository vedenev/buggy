#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 08:50:27 2020

@author: vedenev
"""

import numpy as np
import matplotlib.pyplot as plt

TARGET_DIRECTION = np.asarray([1.0, 0.0])
INITIAL_ANLGE = np.pi/6
INITIAL_POSITION_X = 0.0
INITIAL_POSITION_Y = 0.0

R_MIN  = 25.0

N_STEPS = 3

dt = 0.01
STEP_TIME = 0.3


k_max = 1 / R_MIN

times = np.arange(0, STEP_TIME + dt, dt)



state = np.asarray([INITIAL_POSITION_X, INITIAL_POSITION_Y, INITIAL_ANLGE])
state_history = np.copy(state).reshape(1, 3)
state_history_detailed = np.copy(state).reshape(1, 3)


for step_index in range(N_STEPS):
    
    k = k_max
    v = 100.0
    
    dL = v * dt
    dgamma = k * dL
    for time_index in range(times.size):
        state[0] += dL * np.cos(state[2])
        state[1] += dL * np.sin(state[2])
        state[2] += dgamma
        state_history_detailed = np.concatenate((state_history_detailed, state.reshape(1, 3)), axis=0)
    state_history = np.concatenate((state_history, state.reshape(1, 3)), axis=0)

plt.plot(state_history_detailed[:, 0], state_history_detailed[:, 1], 'k-')
plt.plot(state_history[:, 0], state_history[:, 1], 'r.')
plt.axis('equal')