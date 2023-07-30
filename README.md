# KPI Traces Analysis for E-Marketing Evaluation

This repository contains code for analyzing Key Performance Indicator (KPI) traces in e-marketing. In e-marketing, the evaluation of digital advertising is done in terms of cost and benefit tradeoff. The KPIs, which represent the benefits of a marketing campaign, are quantified in terms of clicks and/or sales (conversions). In this analysis, we will focus on three main KPIs: Cost, Clicks, and Conversions.

## Code Overview

1. Importing Libraries: The code starts by importing necessary Python libraries for data analysis and visualization.

2. Adjusting Plotting Settings: The code sets up plotting configurations, including color schemes and background colors, for better visualization.

3. Importing Data: The code reads the data from the specified CSV file into a NumPy array.

4. Separating KPI Traces: The data is separated into different KPI traces for Cost, Clicks, and Conversions.

5. Finding Useful Companies: The code calculates the correlation between each control trace and the interested trace (the KPI track of interest). The results are plotted using a heatmap, and useful company tracks are identified based on a predefined correlation threshold.

6. Applying Filter to the Data: The code applies a rolling mean filter to smooth the data for each KPI trace.

7. Estimating Expectation and Autocorrelation Matrix: The code defines functions to estimate the expectation and autocorrelation matrix for given KPI traces.

8. Constructing R Matrix: The code constructs the R matrix based on autocorrelation matrices for different KPI traces.

9. Constructing CX Matrix: The code constructs the CX matrix based on the correlation between the KPI track of interest and other control KPI traces.

10. Estimation and Prediction: The code estimates the A vector using the constructed R and CX matrices. Then, it predicts the future values of the KPI track of interest and plots the results.

## Usage

Please note that the code contains different variations for analyzing KPI traces, and you can choose the relevant sections based on your analysis requirements.

## Data Description

The dataset consists of T time samples in days, where T represents the total number of days under observation. Each KPI track (Cost, Clicks, Conversions) for the K companies is stored in a matrix X with dimensions T × K (where K = 20). Each column represents the individual evolution of one KPI track for the k-th company. The KPI tracks are affected by seasonal activity, and their behavior may change due to marketing instances occurring at specific time points Tk. The behavior of each track xk(t) is a combination of the seasonal component x ̄k(t) and the causal signature gk(t - Tk). The causal signature is linearly growing within a certain time interval after the marketing instance and then dropping to zero.

## Analyzing the Data

The primary goal of this analysis is to consider the KPI track of interest (test time series) and compare it with other control time series (control KPIs). The control KPIs are used to predict the behavior of the test time series to evaluate the effectiveness of advertising campaigns.


