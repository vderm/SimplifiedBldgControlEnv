#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Vasken Dermardiros


"""

# Import dependencies
import numpy as np
import pandas as pd

def load_env_data(resample='1h', weather_file="CAN_ON_Ottawa.716280_CWEC.epw"):
    """Create a matrix of time related features. Read part of the EneryPlus weather
    file. Put everything in a Pandas DataFrame and resample to the desired time
    interval.

    Returns a Pandas DataFrame, Index is a TimeSeries.

    """
    # Weather file to use
    from os.path import dirname
    weather_path = dirname(__file__) + "/WeatherData/" + weather_file

    # Adding time-based structure for "vanilla" (multi-layer perceptron) neural net
    # Hours in day, day progression, day symmetry (for 1 day)
    hours_prog = np.linspace(-1,1,24)
    hours_symm = np.concatenate((np.linspace(-1,1,13), np.linspace(1,-1,13)[1:-1]), axis=0)
    hours_dynt = np.concatenate((np.ones(12,), -np.ones(12,)), axis=0)
    # (for 1 year)
    hours_prog = np.tile(hours_prog, 365)
    hours_symm = np.tile(hours_symm, 365)
    hours_dynt = np.tile(hours_dynt, 365)

    # Days (1 for given weekday, 0 otherwise; for 1 week)
    day = np.concatenate((np.ones(24), np.zeros(24*6)), axis=0)
    days = np.zeros((24*7, 7))
    for i in range(7):
        days[:,i] = np.roll(day, 24*i)
    # (for 1 year)
    days = np.tile(days, (52+1,1)) # overextend to cover 365 days
    days = days[:8760, :]          # remove extra bits
    days = days.T

    # Weeks (for 1 year)
    weeks_symm = np.concatenate((np.linspace(-1,1,27), np.linspace(1,-1,27)[1:-1]), axis=0)
    weeks_symm = np.repeat(weeks_symm, 7)
    weeks_symm = np.append(weeks_symm, weeks_symm[-1])
    weeks_symm = np.repeat(weeks_symm, 24)
    weeks_symm = np.roll(weeks_symm, 7*24*7) # roll by 7 weeks, make coldest February week "-1"

    # Putting the hour/day/week stuff all together {dims: (parameter x 8760)}
    # NOTE: Shape: features x time
    X_time = np.concatenate((
        hours_prog.reshape(1,-1),
        hours_symm.reshape(1,-1),
        hours_dynt.reshape(1,-1),
        days, # 7 dims
        weeks_symm.reshape(1,-1)), axis=0)

    # This part reads the weather file. Only using the following data:
    # Weather, get T_ambient, dew point T, solar rad horizontal total
    # X_weather = np.loadtxt(weather_path, delimiter=",", skiprows=8, usecols=[6, 7, 13]).T
    X_weather = np.loadtxt(weather_path, delimiter=",", skiprows=8, usecols=[6, 13]).T

    # Combine them
    X_time_weather = np.concatenate((X_time, X_weather))

    # Create pandas dataframe
    df = pd.DataFrame(X_time_weather.T)
    # df.rename(columns={11: "T_ambient", 12: "T_dew_point", 13: "Solar_rad_horiz"}, inplace=True)
    df.rename(columns={11: "T_ambient", 12: "Solar_rad_horiz"}, inplace=True)
    # Make sure it's not a leap year
    df.index = pd.date_range(start='1/1/2018', end='12/31/2018 23:59:59', freq='1h')

    # Resample, interpolate then ship it! The very last few are off, but it's OK
    resampled_ix = pd.date_range(start='1/1/2018', end='12/31/2018 23:59:59', freq=resample)
    df = df.reindex(resampled_ix)
    return df.interpolate()

def return_env_data(df, how='random', length_days=7, extension_seconds=0):
    """Iterate over the loaded time and weather data.

    how:
    + random: give a random 1 week (or other specified length) worth of data
      starting from a random time.
    + ordered: give a 1 week (or other length) worth of data per week
      (or shift by 1 day?)
    + 'day' (int): give the week (or other length) following the given day, repeated

    """

    # Determine timestep then get total timesteps for specified 'length'
    dt = (df.index[1] - df.index[0]).seconds
    extension_seconds = int(extension_seconds)
    # if length == '1week':  nt = (60*60*24*7+extension_seconds) / dt
    # elif length == '1day': nt = (60*60*24+extension_seconds) / dt
    # elif length == '2day': nt = (60*60*24*2+extension_seconds) / dt
    # elif length == '4day': nt = (60*60*24*4+extension_seconds) / dt
    # else: print("Unknown length specified!")
    nt = (60*60*24*int(length_days)+extension_seconds) / dt

    if how == 'random': start = np.random.randint(0, df.values.shape[0])
    elif how == 'ordered': start = 0
    elif isinstance(how, int): start = how*(60*60*24*7+extension_seconds)/dt
    else: print("'how' is not understood.")
    start, nt = int(start), int(nt)

    indices = range(start, start+nt)
    return df.values.take(indices, axis=0, mode='wrap')

def iterate_env_data(df, how='random', length='1week'):
    # Determine timestep then get total timesteps for specified 'length'
    dt = (df.index[1] - df.index[0]).seconds
    if length == '1week':  nt = (60*60*24*7) / dt
    elif length == '1day': nt = (60*60*24) / dt
    elif length == '2day': nt = (60*60*24*2) / dt
    else: print("Unknown length specified!")
    nt = int(nt)

    indices = np.arange(df.values.shape[0]/nt)
    if how == 'random': np.random.shuffle(indices)
    elif how == 'ordered': pass
    elif isinstance(how, int): indices = np.full_like(indices, how*(60*60*24*7/dt))
    else: print("'how' is not understood.")

    for start_idx in indices:
        start_idx = int(start_idx)
        yield df.values.take(range(start_idx*nt,(start_idx+1)*nt), axis=0, mode='wrap')

# Testing
if __name__ == '__main__':

    df = load_env_data('15min')
    i = 0
    for batch in iterate_env_data(df, how='random', length='1day'):
        # print(batch)
        print(i)
        i += 1

    # # Plot all of them
    # idx_time = [0,1,2,3,4,5,6,7,8,9,10]
    # idx_weather = ['T_ambient', 'T_dew_point', 'Solar_rad_horiz']
    # month_to_plot = 2

    # import matplotlib.pyplot as plt

    # # Hourly Data
    # df = load_env_data()
    # # Time
    # df[idx_time].plot(figsize=(20,20))
    # plt.savefig("./Figs/time_hourly.pdf")
    # df[df.index.month==month_to_plot][idx_time].plot(figsize=(20,20))
    # plt.savefig("./Figs/time_hourly_zoomed.pdf")
    # # Weather
    # df[idx_weather].plot(figsize=(20,20), subplots=True)
    # plt.savefig("./Figs/weather_hourly.pdf")
    # df[df.index.month==month_to_plot][idx_weather].plot(figsize=(20,20), subplots=True)
    # plt.savefig("./Figs/weather_hourly_zoomed.pdf")
    # print("Done with hourly plots.")

    # # Subhourly Data
    # df2 = load_env_data('15min')
    # # Time
    # df2[idx_time].plot(figsize=(20,20))
    # plt.savefig("./Figs/time_subhourly.pdf")
    # df2[df.index.month==month_to_plot][idx_time].plot(figsize=(20,20))
    # plt.savefig("./Figs/time_subhourly_zoomed.pdf")
    # # Weather
    # df2[idx_weather].plot(figsize=(20,20), subplots=True)
    # plt.savefig("./Figs/weather_subhourly.pdf")
    # df2[df.index.month==month_to_plot][idx_weather].plot(figsize=(20,20), subplots=True)
    # plt.savefig("./Figs/weather_subhourly_zoomed.pdf")
    # print("Done with subhourly plots.")
