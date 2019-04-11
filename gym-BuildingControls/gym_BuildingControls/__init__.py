from gym.envs.registration import register

register(
    id='BuildingControls-v0',
    entry_point='gym_BuildingControls.envs:BuildingControlsEnv',
)
