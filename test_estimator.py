# %% Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
import seaborn as sn

# %% Adjust Plotting Settings

colors = ['#BD472A', "#9E2A2B", '#3B3A4A', '#D9D9D6']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
mpl.rcParams['axes.facecolor'] = '202028'

# %% Import Data

data = np.array(pd.read_csv('/Users/Kiarash/Desktop/Drive D/Apply/Msc @ Polimi/Telecommunication Eng/3rd Semester/ADSP/HomeWorks/HW3/presentation/google.csv'))

# %% Separating the KPI Traces

control = [data[:, 0:20], data[:, 20:40], data[:, 40:60]]
names = ['Cost', 'Click', 'Conversion']

# %% Find the useful companies for tracking the interested company (Company #20)

comparison = np.zeros(control[0].shape[1])

rolling_filter_length = 20  # Empirical value based on Engineering choice
smart_bidding_initialization = 397  # The instance time from data is T20 = 397 sample.
correlation_threshold = 0.2  # Empirical value based on Engineering choice

KPI_name = 0  # KPI_name = 0 means 'Cost', KPI_name = 1 means 'Click', KPI_name = 2 means 'Conversion'
useful_company_tracks = []

for kp in control:
    for i in range(kp.shape[1]):
        # Control trace to compare
        input_arr_1 = pd.Series(kp[:, i][:smart_bidding_initialization]).rolling(rolling_filter_length).mean()[
                      rolling_filter_length:]

        # Interested trace (Company #20)
        input_arr_2 = pd.Series(kp[:, 19][:smart_bidding_initialization]).rolling(rolling_filter_length).mean()[
                      rolling_filter_length:]

        # Measure the linear relationship between the arbitrary control trace and the interested trace
        comparison[i] = pearsonr(input_arr_1, input_arr_2)[0]

    # Plotting the comparison results

    # As given in the text, only the relationship between each column and the last column (Column #19) is investigated.
    comparison_DataFrame = pd.DataFrame(data=comparison, index=[str(i) for i in range(kp.shape[1])], columns=[str(19)])

    plt.figure()
    plt.title(names[KPI_name])
    sn.heatmap(comparison_DataFrame, annot=True)

    # Define which company tracks will be useful based on the predefined Correlation Threshold
    useful_company_tracks.append(np.argwhere(abs(comparison) > correlation_threshold))
    print('KPI Name:', names[KPI_name], '\nUseful Company Tracks:\n',
          np.argwhere(abs(comparison) > correlation_threshold))

    KPI_name = KPI_name + 1

# %% Apply Filter to the Data

# data.shape    : (524,60)
# data.shape[0] : 524
# data.shape[1] : 60

ls = [data.copy()]

for i in range(data.shape[1]):
    dum = ls[0][:, i]
    ls[0][:, i] = pd.Series(dum).rolling(7).mean().fillna(0)

control = ls

# Ex. Plot trace 1.
plt.plot(data[:, 0], label='Original')
plt.plot(control[0][:, 0], label='Filtered')
plt.title('Trace 1')
plt.legend()
# leg = plt.gca().get_legend()
# leg.texts[0].set_color('white')
# leg.texts[1].set_color('white')
plt.gca().get_legend().texts[0].set_color('white')
plt.gca().get_legend().texts[1].set_color('white')


# %% Define the function: estimate_expectation

def estimate_expectation(trace1, trace2, lag):
    trace1_ = trace1
    trace2_ = trace2

    c = 0
    count = 0

    smart_bidding_point = 397

    end = smart_bidding_point
    # end = trace1_.shape[0]  # If you want to use the whole data, active this.

    for n in range(0, (end - abs(lag)), 1):
        c = c + trace1_[n] * trace2_[n + abs(lag)]
        count = count + 1
        # print(n)

    c = c / count

    return c


# %% Define the function: estimate_autocorr_mtx

def estimate_autocorr_mtx(trace1, trace2, size):
    trace1_ = trace1
    trace2_ = trace2

    filter_length = size

    acf = []

    for i in range(filter_length):
        acf.append(estimate_expectation(trace1=trace1_, trace2=trace2_, lag=i))

    Rx = np.zeros((filter_length, filter_length))

    for i in range(filter_length):
        for j in range(filter_length):
            Rx[i, j] = acf[abs(i - j)]

    return Rx


# %% Define the function: construct_R_mtx

def construct_R_mtx(indexes, size, column):
    for i in range(len(indexes)):
        for j in range(len(indexes)):
            dum = estimate_autocorr_mtx(trace1=control[column][:, indexes[i]], trace2=control[column][:, indexes[j]],
                                        size=size)
            if j == 0:
                Rx = dum
            else:
                Rx = np.concatenate((Rx, dum), axis=1)

        if i == 0:
            R = Rx
        else:
            R = np.concatenate((R, Rx), axis=0)

    return R


# %% Define the function:

def construct_cx_mtx(indexes, window, causal, column, estimation_index, lag_of_input, lag_of_estimation_index):
    p_vec = []

    for i in range(len(indexes)):

        trace_interested = control[0][:, estimation_index]
        trace_arbitrary = control[0][:, indexes[i]]

        for j in range(-causal, -causal + window):
            if i == 0:
                p_vec.append(estimate_expectation(trace1=trace_interested, trace2=trace_arbitrary,
                                                  lag=(j - lag_of_input - lag_of_estimation_index)))
            else:
                p_vec.append(
                    estimate_expectation(trace1=trace_interested, trace2=trace_arbitrary, lag=(j - lag_of_input)))

    p_vec = np.array(p_vec).reshape(len(p_vec), 1)  # Convert the column vector into a row vector

    return p_vec


# %% Defining

# All KPI traces
# cost = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # indexes: Cost
# click = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # indexes: Click
# conversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # indexes: Conversion

# Correlated traces
# cost = [1, 2, 9, 12, 13, 14, 15, 16, 17, 18]  # indexes: Cost
# click = [0, 2, 6, 7, 12, 14, 15, 16, 17, 18]  # indexes: Click
# conversion = [2, 6, 7, 8, 11, 14, 15, 17, 18]  # indexes: Conversion

# Question 1- Used all traces of Conversion KPI
cost = []  # indexes: Cost
click = []  # indexes: Click
conversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # indexes: Conversion
window = 10
causal = 9

# Question 2a(Only conversion)- used all traces of Conversion(except 18 due to outlier, 19 due to problem definition)
# cost = []  # indexes: Cost
# click = []  # indexes: Click
# conversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # indexes: Conversion
# window = 10
# causal = 9

# Question 2b & 2c(Only conversion) - used useful traces of Conversion(except 18 due to outlier, 19 due to problem definition)
# cost = []  # indexes: Cost
# cost = []  # indexes: Cost
# click = []  # indexes: Click
# conversion = [2, 6, 7, 8, 11, 14, 15, 17]  # indexes: Conversion
# window = 10
# causal = 9

# Question 3a(All KPIs)  used all traces of all KPIs (except 18 due to outlier, 19 due to problem definition)
# cost = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # indexes: Cost
# click = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # indexes: Click
# conversion = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # indexes: Conversion
# window = 10
# causal = 9

# Question 3b & 3c(All KPIs) - used useful traces of all KPIs (except 18 due to outlier, 19 due to problem definition)
# cost = [1, 2, 9, 12, 13, 14, 15, 16, 17]  # indexes: Cost
# click = [0, 2, 6, 7, 12, 14, 15, 16, 17]  # indexes: Click
# conversion = [2, 6, 7, 8, 11, 14, 15, 17]  # indexes: Conversion
# window = 10
# causal = 9

# All KPIs - Question 4a
# cost = [1, 2, 9, 12, 13, 14, 15, 16, 17]  # indexes: Cost
# click = [0, 2, 6, 7, 12, 14, 15, 16, 17]  # indexes: Click
# conversion = [2, 6, 7, 8, 11, 14, 15, 17]  # indexes: Conversion
# window = 9
# causal = 0


# indexes_cost = cost
# indexes_click = [(idx+20) for idx in click]
# indexes_conversion = [(idx+40) for idx in conversion]
# indexes = indexes_cost + indexes_click + indexes_conversion
indexes = cost + [value + 20 for value in click] + [value + 40 for value in conversion]

estimation_index = 59  # Index number to be estimated

lag_of_input = 0
lag_of_estimation_index = 0

smoothing_window = 3

column = 0

# Estimator

R_matrix = construct_R_mtx(indexes, window, column)
p_vector = construct_cx_mtx(indexes, window, causal, column, estimation_index, lag_of_input, lag_of_estimation_index)

A = np.linalg.inv(R_matrix) @ p_vector

# Prediction

# Step-Size depends on the:
#   lag_of_input
#   lag_of_estimation_index

start = causal + lag_of_input + lag_of_estimation_index
end = len(control[column][:, estimation_index]) - window + causal

prediction = []
prediction_error = []
plotting_index = []

lag = causal + lag_of_input

for i in range(start, end): #9 to 514(523 - window(=9))

    for j in range(len(indexes)):

        tmp = control[column][i - lag:i - lag + window, indexes[j]].reshape(window, 1) # 9-9: 9-9+10

        if j == 0:
            x_tilda = tmp
        else:
            x_tilda = np.concatenate((x_tilda, tmp))

    y_predict = A.transpose() @ x_tilda
    prediction.append(y_predict)

    MSE = abs(control[column][i, estimation_index] - y_predict) ** 2
    prediction_error.append(MSE)

    plotting_index.append(i)

# Filtering the prediction

prediction = pd.Series(np.array(prediction).reshape(len(prediction)))
prediction_filtered = prediction.rolling(smoothing_window, center=True).mean().fillna(method="ffill")

prediction_error = pd.Series(np.array(prediction_error).reshape(len(prediction_error)))
prediction_error_filtered = prediction_error.rolling(10, center=True).mean().fillna(method="ffill")

# Plotting

plt.figure()
plt.plot(plotting_index, prediction_filtered, label="Prediction")
plt.plot(control[column][:, estimation_index], label="Original")

plt.vlines(x=397, ymin=min(control[column][:, estimation_index]),
           ymax=max(control[column][:, estimation_index]), colors='green', ls=':', lw=5,
           label="Smart Bidding Point")

plt.legend()
plt.gca().get_legend().texts[0].set_color('white')
plt.gca().get_legend().texts[1].set_color('white')
plt.gca().get_legend().texts[2].set_color('white')

plt.figure()
plt.plot(plotting_index, prediction_error_filtered, label="Error")

plt.vlines(x=397, ymin=min(control[column][:, estimation_index]),
           ymax=max(control[column][:, estimation_index]), colors='green', ls=':', lw=5,
           label="Smart Bidding Point")

plt.legend()
plt.gca().get_legend().texts[0].set_color('white')
plt.gca().get_legend().texts[1].set_color('white')

# %% Q5
# plt.plot(control[column][:, 59]/control[column][:, 19])
# plt.legend()
# plt.gca().get_legend().texts[0].set_color('white')
plt.plot(control[0][:,59]/control[0][:,19])
