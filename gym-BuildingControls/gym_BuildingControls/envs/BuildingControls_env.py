#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Created on Apr 2018
Started to make into an OpenAI gym Jun 21 2018
Cleaned version to upload following IBPSA 2019 Rome Publication April 2019

@author: Vasken Dermardiros

Building environment to be use in an RL setting similar to OpenAI Gym.

User inputs settings to create the building using a resistance capacitance (RC)
thermal network. User also specifies the temperature profiles of day; the class
will 'decide' which information to supply to the agent. Noise can be added to
weather predictions for added realism.

TODO
+ Setpoints change with time -> new occupant

EXTRA
+ Debug code: import pdb; pdb.set_trace()

"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding

# fn_sim     : helper functions for simulating the environment
# fn_env     : helper functions for modifying the environment, reading ambient temperature from weather file, applying noise, shuffling order the agent sees
# bldg_models: buldings models, 2 simple models and the general model
from gym_BuildingControls.envs import bldg_models, fn_sim, fn_env

import numpy as np

class BuildingControlsEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        ''' Define the building controls environment.'''
        self.curr_episode = -1 # keep track of episode to change curriculum over time
        # NOTE: Following vector to specify which training mode will be run for how many episodes; use '1' to practically skip that mode
        # NOTE: Training too much on simple cases results in a unrecoverable collapse of the agent; to avoid!
        # self.curriculum = [5, 5, 5, 5, 50, 500, 1000, 10000, 1e100, 1e100, 1e100, 1e100,] # episodes per perturbation mode, in order, "last mode" goes on for total episodes to run
        self.curriculum = [50, 50, 50, 50, 100, 500, 1000, 10000, 1e100, 1e100, 1e100, 1e100,] # episodes per perturbation mode, in order, "last mode" goes on for total episodes to run
        self.bldg = 0 # 1st-order model: direct conditionning of room air
        # self.bldg = 1 # 2nd-order model: radiant slab system to condition room air node
        self._get_building_def()
        # Weather prediction steps to consider, number of them and how far ahead
        self.weather_pred_steps = [0, 15, 30, 60, 180, 240, 480] # minutes, in ascending order
        self.weather_pred_uncert_std = [0., 0.05, 0.05, 0.05, 0.05, 0.07, 0.10] # standard deviation, temp, from statistical study
        self.perturbation_mode = 0
        self.change_perturbations_mode()
        # settings for 1st-order model, otherwise settings for other models
        self.settings_mode = 0 if (self.bldg == 0) else 1
        self.change_settings_mode()

        # What the agent can do
        self.nA = 3 # -/0/+
        self.action_space = spaces.Discrete(self.nA)

        # What the environment keeps track of
        self.reset()

        # What the agent sees
        self.nS = len(self.state)
        low = np.hstack((
            0.,
            self.minA,
            -1.,
            0.,
            -1.*np.ones(self.timeindicator.shape[1]),
            -40.*np.ones(len(self.weather_pred_steps)),
            -20,
        ))
        high = np.hstack((
            45.,
            self.maxA,
            1.,
            self.nT,
            1.*np.ones(self.timeindicator.shape[1]),
            50.*np.ones(len(self.weather_pred_steps)),
            20,
        ))
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # For OpenAI Baseline A2C and PPO2 frameworks
        self.nenvs = self.num_envs = 1


    def get_state(self):
        self.state = np.hstack((
            self.T[self.T_sensor_node],  # room temperature, where sensor is located
            self.heat_cool_s, # heating cooling state
            self.comfort,
            self.nT - self.t, # time left for episode
            self.timeindicator[self.t,], # NOTE: length 11
            np.array([
                self.TK[self.t+int(self.weather_pred_steps[i]/self.timestep)] + \
                np.random.normal(0,self.weather_pred_uncert_std[i]) \
                for i in range(len(self.weather_pred_steps))
            ]).flatten(), # NOTE: length = len(weather_pred_steps)
            # TODO: add building type as an input: {light, heavy, office}
            self.T[self.T_sensor_node] - 22.5,  # room temperature difference vs 22.5
        ))
        return self.state


    def reset(self):
        # Curriculum
        self.curr_episode += 1
        if (self.curr_episode > self.curriculum[self.perturbation_mode]):
            print("Completed lesson. Incrementing perturbation mode.")
            self.perturbation_mode += 1
            self.change_perturbations_mode()
            self.curr_episode = 0
        else:
            self._get_perturbations()

        self.t = 0
        self.T = np.random.normal(22.5, scale=self.T_start_std, size=self.nN) # initial temperatures
        self.heat_cool_s = self.heat_cool_off # start with HVAC off
        self.comfort = 0

        return self.get_state()


    def step(self, action):
        ''' One step in the world.

        [observation, reward, done = env.step(action)]
        action: which heating/cooling setting to use

        @return observation: state temperatures
        @return reward: the reward associated with the next state
        @return done: True if the state is terminal
        '''
        done = False
        reward = 0.
        self.comfort = 0

        # If action is passed as a numpy array instead of a scalar
        if isinstance(action, np.ndarray): action = action[0]

        # Action space
        self.heat_cool_s += (action-1) # to have [0,1,2] -> [-1,0,+1]
        self.heat_cool_s = np.max((self.heat_cool_s, self.minA)) # lower action bound
        self.heat_cool_s = np.min((self.heat_cool_s, self.maxA)) # upper action bound

        Q_applied = self.heat_cool_levels[self.heat_cool_s]['power']
        self.T = self.calculate_next_T(Q_applied) # for all thermal loads
        Tr = self.T[self.T_sensor_node] # temperature at sensor location

        # HVAC energy cost
        reward += self.cost_factor * self.heat_cool_levels[self.heat_cool_s]['cost']

        if self.comfort_penalty_scheme == 'power':
            # Thermal comfort: Using an exponential penalty based on distance from comfort bounds,
            # no terminal penalty at an extreme temp, smoother penalty
            # from RL paper: Deep Reinforcement Learning for Optimal Control of Space Heating, Nagy Kazmi et al, eq. 4
            if (Tr < self.T_low_sp): # too cold
                reward += -4*1.35**(self.T_low_sp - Tr)
                self.comfort = -1
            elif (Tr > self.T_high_sp): # too hot
                reward += -3*1.30**(Tr - self.T_high_sp)
                self.comfort = 1
            # else:
                # reward += 0

        elif self.comfort_penalty_scheme == 'linear':
            if (Tr < self.T_low_sp): # too cold
                reward -= (self.T_low_sp - Tr)
                self.comfort = -1
            elif (Tr > self.T_high_sp): # too hot
                reward -= (Tr - self.T_high_sp)
                self.comfort = 1
            # else:
                # reward += 0

        elif self.comfort_penalty_scheme == 'linear with termination':
            # NOTE: the following method has been depreciated in favour of a smoother penalty scheme
            if (Tr < self.T_low_limit) or (Tr > self.T_high_limit):
                reward += self.penalty_limit_factor*self.nT
                done = True
            elif (Tr < self.T_low_sp) or (Tr > self.T_high_sp):
                reward += self.penalty_sp
                done = False
            # else:
                # reward += 0
                # done = False

            # You're hot (1) then you're cold (-1), ok (0)
            if (Tr > self.T_high_sp):  self.comfort = 1
            elif (Tr < self.T_low_sp): self.comfort = -1
            else: self.comfort = 0

        # Excessive toggling
        reward += self.cost_factor * self.penalty_hvac_toggle*np.abs(action-1)

        # Increment time
        self.t += 1
        if self.t >= (self.nT-1):
            reward += self.reward_termination # Looks like we made it!
            done = True

        return self.get_state(), reward, done, {'T':self.T}


    def render(self, mode='human', close=False):
        return


    # Calculate temperatures for next timesteps: T(t+1) =  Q(t) * U^-1
    # Q-vector: Q = Qhvac + Qin + F*TK(t+1) + C/dt*T(t)
    def calculate_next_T(self, Q_applied):
        Q_applied_1hot = np.eye(1,self.nN,self.heat_cool_node).flatten() * Q_applied
        Q = Q_applied_1hot + self.Q[self.t] + np.dot(self.F,self.TK[self.t+1]) + np.multiply(self.C.T/self.dt, self.T).flatten()
        return np.dot(Q, self.U_inv)


    def change_perturbations_mode(self):
        self.perturbation_loaded = False # Flag to keep track if perturbation weather file has been loaded
        self._get_perturbations()
        print("Perturbation mode: %s" % self.perturbation_mode)


    def change_settings_mode(self):
        self._get_settings()
        print("Settings mode: %s" % self.settings_mode)


    # Building model. For more details, please consult the "bldg_models" file.
    def _get_building_def(self):
        '''
        building_def contains U, F, C matrices
          nN: number of interior nodes
          nM: number of boundary nodes
          (U: how thermal nodes are connected to each other [nN x nN])
          U_inv: thermal node matrix for implicit finite difference calculation [nN x nN]
          F:  how thermal nodes are connected to boundaries [nN x nM]
          C:  thermal capacitances at nodes [nN]
          dt: time steps, in seconds
          T_start: temperature of room nodes at the start [nN]
          T_sensor_node: where the thermostat is located
          heat_cool_levels: dictionary map actions to heat/cool outputs with cost penalty
          heat_cool_node: where to apply the heating/cooling
        '''
        self.timestep = 15 # minutes
        # self.timestep = 5 # minutes
        self.dt = self.timestep*60. # timestep in seconds

        if self.bldg == 0:
            ''' Typical single family house in Montreal, heating delivered to the space directly.
                1st-order RC model with effective overall capactiance and resistance values. '''
            print("Building: 0, 1st-order model.")
            self.U_inv, self.F, self.C, self.nN, self.nM = bldg_models.mF1C1(F_in=250, C_in=12e6, dt=self.dt) # 1-node thermal network model
            self.Q_solar_fraction = 0.5
            self.T_sensor_node = 0 # thermostat measurement location node
            self.heat_cool_node = 0 # heating/cooling applied to this thermal node

            self.heat_cool_levels = {
                # NOTE: must be in ascending order, no limitation on number
                # 0: {'power':  -5000., 'cost': -2.,}, # power in watts (-ve is cooling)
                # 1: {'power':  -2500., 'cost': -1.,},
                0: {'power':  -3000., 'cost': -2.,}, # power in watts (-ve is cooling)
                1: {'power':      0., 'cost':  0.,},
                2: {'power':   5000., 'cost': -2.,},
                3: {'power':  10000., 'cost': -4.,},
                # 3: {'power':   2500., 'cost': -1.,},
                # 4: {'power':   5000., 'cost': -2.,},
                # 5: {'power':   7500., 'cost': -3.,},
                # 6: {'power':  10000., 'cost': -4.,},
            }
        elif self.bldg == 1:
            ''' Space where heating/cooling is applied on a thermal node not the same as the
                temperature sensor node such as the case of a radiant slab-based conditionning
                system. Here,
                    U_in: conductor between the air node and slab node;
                    F_in: conductor between the air node and the ambient node;
                    C_in: room air capacitance;
                    C_slab: slab capacitance; '''
            print("Building: 1, 2nd-order model.")
            self.U_inv, self.F, self.C, self.nN, self.nM = bldg_models.mU1F1C2(U_in=15.*185., F_in=250., C_in=2e6, C_slab=10e6, dt=self.dt) # 2-node thermal network model
            self.Q_solar_fraction = 0.1
            self.T_sensor_node = 0 # thermostat measurement location node (air)
            self.heat_cool_node = 1 # heating/cooling applied to this thermal node (slab)

            self.heat_cool_levels = {
                # NOTE: must be in ascending order, no limitation on number
                0: {'power': -3000., 'cost': -2.,}, # power in watts (-ve is cooling)
                1: {'power':     0., 'cost':  0.,},
                2: {'power':  5000., 'cost': -2.,},
                3: {'power': 10000., 'cost': -4.,},
            }
        else:
            print("Unknown model specified!")

        # self.heat_cool_off = int((self.maxA-self.minA)/2) Old way
        for key, val in self.heat_cool_levels.items():
            if val['power'] == 0: self.heat_cool_off = key
        keys = [j for i,j in enumerate(self.heat_cool_levels.keys())]
        self.minA = min(keys)
        self.maxA = max(keys)


    # Perturbations to the model
    def _get_perturbations(self):
        '''
        perturbations contains TK, Q matrices
          nT: number of timesteps total
          TK: temperatures at boundaries [nT x nM]
          Q:  heat input into interior nodes [nT x nN]
        '''
        # synthetic case 0
        # fixed comfortable temp, no baseload
        if (self.perturbation_mode == 0) and (not self.perturbation_loaded):
            self.T_start_std = 0. # spread on initial temperatures
            self.nT = 180 # total number of timesteps including t_0 initial condition, T(t+1) depends on TK(t+1)
            self.timeindicator = np.zeros((self.nT, 11))
            lenTK = int(self.nT + self.weather_pred_steps[-1]/self.timestep)
            self.TK = fn_sim.periodic(22.5, 0., 15, 86400., self.dt, lenTK)[:,np.newaxis]
            Q_solar = fn_sim.halfperiodic(0., 12., 86400., self.dt, self.nT)[:,np.newaxis]
            Q_baseload = 0.
            self.Q = Q_solar + Q_baseload
            self.perturbation_loaded = True
            print("Loaded perturbations.")

        # synthetic case 1
        # fixed comfortable temperature, fixed baseload = hvac setting
        if (self.perturbation_mode == 1) and (not self.perturbation_loaded):
            self.T_start_std = 0. # spread on initial temperatures
            self.nT = 180 # total number of timesteps including t_0 initial condition, T(t+1) depends on TK(t+1)
            self.timeindicator = np.zeros((self.nT, 11))
            lenTK = int(self.nT + self.weather_pred_steps[-1]/self.timestep)
            self.TK = fn_sim.periodic(22.5, 0., 15, 86400., self.dt, lenTK)[:,np.newaxis]
            #self.TK = fn_sim.random_TK(dt, nT+6).reshape(nT+6,1)
            Q_solar = fn_sim.halfperiodic(0., 12., 86400., self.dt, self.nT)[:,np.newaxis]
            Q_baseload = 500.
            self.Q = Q_solar + Q_baseload
            self.perturbation_loaded = True
            print("Loaded perturbations.")

        # synthetic case 2
        # temperature changes periodically, fixed baseload = hvac setting, longer time
        if (self.perturbation_mode == 2) and (not self.perturbation_loaded):
            self.T_start_std = 0.5 # spread on initial temperatures
            self.nT = 360 # total number of timesteps including t_0 initial condition, T(t+1) depends on TK(t+1)
            self.timeindicator = np.zeros((self.nT, 11))
            lenTK = int(self.nT + self.weather_pred_steps[-1]/self.timestep)
            self.TK = fn_sim.periodic(10., 10., 15, 86400., self.dt, lenTK)[:,np.newaxis]
            #self.TK = fn_sim.random_TK(dt, nT+6).reshape(nT+6,1)
            Q_solar = fn_sim.halfperiodic(0., 12., 86400., self.dt, self.nT)[:,np.newaxis] # sunny day, total heat gain through windows
            Q_baseload = 500.
            self.Q = Q_solar + Q_baseload
            self.perturbation_loaded = True
            print("Loaded perturbations.")

        # synthetic case 3
        # temperature changes periodically larger dT, smaller baseload + half-sine solar gains, longer time
        if (self.perturbation_mode == 3) and (not self.perturbation_loaded):
            self.T_start_std = 0.5 # spread on initial temperatures
            self.nT = 360 # total number of timesteps including t_0 initial condition, T(t+1) depends on TK(t+1)
            self.timeindicator = np.zeros((self.nT, 11))
            lenTK = int(self.nT + self.weather_pred_steps[-1]/self.timestep)
            # self.TK = fn_sim.periodic(20., 15., 15, 86400., self.dt, lenTK).reshape(lenTK,1)
            self.TK = fn_sim.periodic(10., 15., 15, 86400., self.dt, lenTK)[:,np.newaxis]
            #self.TK = fn_sim.random_TK(dt, nT+6).reshape(nT+6,1)
            # Q_solar = fn_sim.halfperiodic(600., 12., 86400., self.dt, self.nT).reshape(self.nT,1) # sunny day, total heat gain through windows
            Q_solar = fn_sim.halfperiodic(600., 12., 86400., self.dt, self.nT)[:,np.newaxis] # sunny day, total heat gain through windows
            Q_baseload = 300.
            self.Q = Q_solar + Q_baseload
            self.perturbation_loaded = True
            print("Loaded perturbations.")

        # realistic case 0: real weather, daily, start on day 80
        if (self.perturbation_mode == 4) and (not self.perturbation_loaded):
            self.T_start_std = 0.5 # spread on initial temperatures
            # NOTE def load_env_data(resample='1h', weather_file="CAN_ON_Ottawa.716280_CWEC.epw")
            self.df_timeweather = fn_env.load_env_data(str(self.timestep)+'min', weather_file="CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw") # XXX: load this once!
            # self.timeweather = fn_env.return_env_data(self.df_timeweather, how=80, length='1day', extension_seconds=60*self.weather_pred_steps[-1])
            self.timeweather = fn_env.return_env_data(self.df_timeweather, how=80, length_days=1, extension_seconds=60*self.weather_pred_steps[-1])
            # self.nT = self.timeweather.shape[0] - 6 # NOTE: cut short depending on what is supplied as predicted to states; otherwise, we will be out of range
            self.nT = int(24*60/self.timestep) # NOTE: cut short depending on what is supplied as predicted to states; otherwise, we will be out of range, 1 day
            self.timeindicator = self.timeweather[:,0:11]
            self.TK = self.timeweather[:,11][:,np.newaxis]
            Q_solar = self.Q_solar_fraction*self.timeweather[:,12][:,np.newaxis]
            Q_baseload = 250.
            self.Q = Q_solar + Q_baseload
            self.perturbation_loaded = True
            print("Loaded perturbations.")

        # realistic case 1: real weather, daily, shuffled
        if self.perturbation_mode == 5:
            self.T_start_std = 1.5 # 0.5 # spread on initial temperatures
            # NOTE def load_env_data(resample='1h', weather_file="CAN_ON_Ottawa.716280_CWEC.epw")
            if (not self.perturbation_loaded):
                self.df_timeweather = fn_env.load_env_data(str(self.timestep)+'min', weather_file="CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw")
                self.perturbation_loaded = True
                print("Loaded perturbations.")
            self.timeweather = fn_env.return_env_data(self.df_timeweather, how='random', length_days=1, extension_seconds=60*self.weather_pred_steps[-1])
            # self.nT = self.timeweather.shape[0] - 6 # NOTE: cut short depending on what is supplied as predicted to states; otherwise, we will be out of range
            self.nT = int(24*60/self.timestep) # NOTE: cut short depending on what is supplied as predicted to states; otherwise, we will be out of range, 1 day
            self.timeindicator = self.timeweather[:,0:11]
            self.TK = self.timeweather[:,11][:,np.newaxis]
            Q_solar = self.Q_solar_fraction*self.timeweather[:,12][:,np.newaxis]
            Q_baseload = np.random.uniform(low=100.,high=800.)
            self.Q = Q_solar + Q_baseload

        # realistic case 1: real weather, 2 days, shuffled
        if self.perturbation_mode == 6:
            self.T_start_std = 2.0 # 0.5 # spread on initial temperatures
            # NOTE def load_env_data(resample='1h', weather_file="CAN_ON_Ottawa.716280_CWEC.epw")
            if (not self.perturbation_loaded):
                self.df_timeweather = fn_env.load_env_data(str(self.timestep)+'min', weather_file="CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw")
                self.perturbation_loaded = True
                print("Loaded perturbations.")
            self.timeweather = fn_env.return_env_data(self.df_timeweather, how='random', length_days=2, extension_seconds=60*self.weather_pred_steps[-1])
            # self.nT = self.timeweather.shape[0] - 6 # NOTE: cut short depending on what is supplied as predicted to states; otherwise, we will be out of range
            self.nT = int(24*60*2/self.timestep) # NOTE: cut short depending on what is supplied as predicted to states; otherwise, we will be out of range, 1 day
            self.timeindicator = self.timeweather[:,0:11]
            self.TK = self.timeweather[:,11][:,np.newaxis]
            Q_solar = self.Q_solar_fraction*self.timeweather[:,12][:,np.newaxis]
            Q_baseload = np.random.uniform(low=100.,high=800.)
            self.Q = Q_solar + Q_baseload

        # realistic case 1: real weather, random length of days [gamma distribution, k=2, theta=1], shuffled
        if self.perturbation_mode == 7:
            self.T_start_std = 2.0 # 0.5 # spread on initial temperatures
            # NOTE def load_env_data(resample='1h', weather_file="CAN_ON_Ottawa.716280_CWEC.epw")
            if (not self.perturbation_loaded):
                self.df_timeweather = fn_env.load_env_data(str(self.timestep)+'min', weather_file="CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw")
                self.perturbation_loaded = True
                print("Loaded perturbations.")
            length_days = int(np.ceil(np.random.gamma(2, 1))) + 1
            self.timeweather = fn_env.return_env_data(self.df_timeweather, how='random', length_days=length_days, extension_seconds=60*self.weather_pred_steps[-1])
            # self.nT = self.timeweather.shape[0] - 6 # NOTE: cut short depending on what is supplied as predicted to states; otherwise, we will be out of range
            self.nT = int(24*60*length_days/self.timestep) # NOTE: cut short depending on what is supplied as predicted to states; otherwise, we will be out of range, 1 day
            self.timeindicator = self.timeweather[:,0:11]
            self.TK = self.timeweather[:,11][:,np.newaxis]
            Q_solar = self.Q_solar_fraction*self.timeweather[:,12][:,np.newaxis]
            Q_baseload = np.random.uniform(low=100.,high=800.)
            self.Q = Q_solar + Q_baseload

        # realistic case 2: real weather, weekly, shuffled
        if self.perturbation_mode == 8:
            self.T_start_std = 2.0 # 0.5 # spread on initial temperatures
            # NOTE def load_env_data(resample='1h', weather_file="CAN_ON_Ottawa.716280_CWEC.epw")
            if (not self.perturbation_loaded):
                self.df_timeweather = fn_env.load_env_data(str(self.timestep)+'min', weather_file="CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw")
                self.perturbation_loaded = True
                print("Loaded perturbations.")
            self.timeweather = fn_env.return_env_data(self.df_timeweather, how='random', length_days=7, extension_seconds=60*self.weather_pred_steps[-1])
            # fix the following to be longer by timestep (or just a day more?) and cut
            self.nT = int((24*60*7)/self.timestep)
            # self.nT = int((24*60*7-self.weather_pred_steps[-1])/self.timestep) # NOTE: cut short depending on what is supplied as predicted to states; otherwise, we will be out of range, 1 week minus predicted weather
            self.timeindicator = self.timeweather[:,0:11]
            self.TK = self.timeweather[:,11][:,np.newaxis]
            Q_solar = self.Q_solar_fraction*self.timeweather[:,12][:,np.newaxis]
            Q_baseload = np.random.uniform(low=100.,high=800.)
            self.Q = Q_solar + Q_baseload

        # realistic case 3: real weather, different than previous, weekly, shuffled
        if self.perturbation_mode == 9:
            self.T_start_std = 0.5 # spread on initial temperatures
            # NOTE def load_env_data(resample='1h', weather_file="CAN_ON_Ottawa.716280_CWEC.epw")
            if (not self.perturbation_loaded):
                self.df_timeweather = fn_env.load_env_data(str(self.timestep)+'min', weather_file="CAN_ON_Toronto.716240_CWEC.epw")
                self.perturbation_loaded = True
                print("Loaded perturbations.")
            self.timeweather = fn_env.return_env_data(self.df_timeweather, how='random', length_days=7, extension_seconds=60*self.weather_pred_steps[-1])
            self.nT = int((24*60*7)/self.timestep)
            self.timeindicator = self.timeweather[:,0:11]
            self.TK = self.timeweather[:,11][:,np.newaxis]
            Q_solar = self.Q_solar_fraction*self.timeweather[:,12][:,np.newaxis]
            Q_baseload = np.random.uniform(low=100.,high=800.)
            self.Q = Q_solar + Q_baseload

        # test case: Montreal winter, real weather, weekly
        if (self.perturbation_mode == 10) and (not self.perturbation_loaded):
            self.T_start_std = 0.5 # spread on initial temperatures
            # NOTE def load_env_data(resample='1h', weather_file="CAN_ON_Ottawa.716280_CWEC.epw")
            self.df_timeweather = fn_env.load_env_data(str(self.timestep)+'min', weather_file="CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw") # XXX: load this once!
            self.timeweather = fn_env.return_env_data(self.df_timeweather, how=2, length_days=7, extension_seconds=60*self.weather_pred_steps[-1])
            self.nT = int((24*60*7)/self.timestep)
            self.timeindicator = self.timeweather[:,0:11]
            self.TK = self.timeweather[:,11][:,np.newaxis]
            Q_solar = self.Q_solar_fraction*self.timeweather[:,12][:,np.newaxis]
            Q_baseload = np.random.uniform(low=100.,high=800.)
            self.Q = Q_solar + Q_baseload
            self.perturbation_loaded = True
            print("Loaded perturbations.")

        # test case: Montreal summer, real weather, weekly
        if (self.perturbation_mode == 11) and (not self.perturbation_loaded):
            self.T_start_std = 0.5 # spread on initial temperatures
            # NOTE def load_env_data(resample='1h', weather_file="CAN_ON_Ottawa.716280_CWEC.epw")
            self.df_timeweather = fn_env.load_env_data(str(self.timestep)+'min', weather_file="CAN_PQ_Montreal.Intl.AP.716270_CWEC.epw") # XXX: load this once!
            self.timeweather = fn_env.return_env_data(self.df_timeweather, how=28, length_days=7, extension_seconds=60*self.weather_pred_steps[-1])
            self.nT = int((24*60*7)/self.timestep)
            self.timeindicator = self.timeweather[:,0:11]
            self.TK = self.timeweather[:,11][:,np.newaxis]
            Q_solar = self.Q_solar_fraction*self.timeweather[:,12][:,np.newaxis]
            Q_baseload = np.random.uniform(low=100.,high=800.)
            self.Q = Q_solar + Q_baseload
            self.perturbation_loaded = True
            print("Loaded perturbations.")

    # Model settings for RL
    def _get_settings(self):
        '''
        settings
          T_high_limit: high-limit temperature that will terminate episode
          T_low_limit: low-limit temperature that will terminate episode
          T_high_sp: high setpoint temperature, comfort violation penalty above
          T_low_sp: low setpoint temperature, comfort violation penalty below
          penalty_hvac_toggle: penalty for action toggling
          comfort_penalty_scheme: how to apply penalties if out of comfort bounds: 'power', 'linear', 'linear with termination'
        '''
        if self.settings_mode == 0:
            self.T_high_sp = 25.     # setpoint temps: if out of range, small constant penalty per timestep
            self.T_low_sp = 20.
            self.reward_termination = 0.
            self.penalty_hvac_toggle = -0.8
            self.cost_factor = 0.2
            self.comfort_penalty_scheme = 'linear'
            self.T_high_limit = 35.  # limit temperatures: if reached, episode terminates with large penalty (used for 'linear with termination' scheme)
            self.T_low_limit = 10.
            self.penalty_limit_factor = -2.
            self.penalty_sp = -1.
        if self.settings_mode == 1:
            self.T_high_sp = 25.     # setpoint temps: if out of range, small constant penalty per timestep
            self.T_low_sp = 20.
            self.reward_termination = 0.
            self.penalty_hvac_toggle = -1.1
            self.cost_factor = 0.2
            self.comfort_penalty_scheme = 'linear'
            self.T_high_limit = 35.  # limit temperatures: if reached, episode terminates with large penalty (used for 'linear with termination' scheme)
            self.T_low_limit = 10.
            self.penalty_limit_factor = -2.
            self.penalty_sp = -1.
