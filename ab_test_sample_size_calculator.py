import streamlit as st
import pandas as pd
from scipy import stats
from scipy.stats import norm
#import statsmodels.stats.api as sms
import plotly.graph_objects as go
import os
from datetime import datetime
import math

# Define the Streamlit app
def main():
    # Set the title of the app
    st.title("AB Test Sample Size for Proportions")

    # Define input fields in the sidebar
    st.sidebar.header("Input Parameters")
    test_name = st.sidebar.text_input("Test Name", 'Some Test on % Submit to Confirm Uplift')
    daily_traffic = st.sidebar.number_input("Daily Traffic (Total)", min_value=1, value=7300)
    p1 = st.sidebar.number_input("Control Group CR (History)", min_value=0.0, max_value=1.0, step=0.01, value=0.10)
    uplift = st.sidebar.number_input("Test CR Uplift (Expected)", min_value=0.0, step=0.01, value=0.10)
    control_share = st.sidebar.number_input("Control Size (%)", min_value=0, max_value=100, step=1, value=50)
    test_share = 100 - control_share  # Calculate test_share based on control_share
    alpha = st.sidebar.number_input("Alpha", min_value=0.0, max_value=1.0, step=0.01, value=0.05)
    beta = st.sidebar.number_input("Beta (Power)", min_value=0.0, max_value=1.0, step=0.01, value=0.2)

    # Calculate sample sizes and duration
    p2 = p1 * (1 + uplift)
    sample_size_control, sample_size_test = calculate_sample_size(p1, p2, control_share, test_share, alpha, beta)
    duration = math.ceil((sample_size_control + sample_size_test) / daily_traffic)

    # Display the summary
#    st.subheader("Summary")
    summary_text = f"""
        {test_name}
        control CR = {p1 * 100:.2f}%
        expeted CR = {p2 * 100:.2f}%
        uplift     = {(p2/p1-1) * 100:.2f}%
        for {control_share}/{test_share} traffic allocation we need to collect
        sample_size_control: {sample_size_control:,}
        sample_size_test:    {sample_size_test:,}
        --------------------------------------------------------------------
        TOTAL SIZE: {sample_size_control + sample_size_test:,}  || Approx. {duration:,} days for daily {daily_traffic:,} users
        --------------------------------------------------------------------
        Only for binominal distibution, two-tailed test
        alpha = {alpha * 100:.2f}%
        power = {(1 - beta) * 100:.2f}%
        """
    st.code(summary_text)

    sample_sizes_control = []
    sample_sizes_test = []
    deltas = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    for delta in deltas:
        p2=p1*(1+delta)
        sample_size_control, sample_size_test = calculate_sample_size(p1, p2, control_share, test_share, alpha=0.05, beta=0.2)
        sample_sizes_control.append(sample_size_control)
        sample_sizes_test.append(sample_size_test)

    sum_list = [control + test for control, test in zip(sample_sizes_control, sample_sizes_test)]
    total_list = [math.ceil(total / daily_traffic) for total in sum_list]


    ## Chart size
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=deltas, y=sample_sizes_control, name='Control Group'))
    fig.add_trace(go.Scatter(x=deltas, y=sample_sizes_test, name='Test Group'))

    fig.update_layout(
        title=f'Minimum required sample size for detectable delta for CR metric for {control_share}/{test_share} traffic allocation',
        xaxis_title='Uplift (Delta)',
        yaxis_title='Sample Size',
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig)


    ## Chart duratiom
    figd = go.Figure()
    figd.add_trace(go.Scatter(x=deltas, y=total_list, name='Duration Days', showlegend=False))

    figd.update_layout(
        title=f'Required days for minimum detectable delta for CR metric for {control_share}/{test_share} traffic allocation',
        xaxis_title='Uplift (Delta)',
        yaxis_title='Days',
        template="plotly_white",
        #legend=dict(x=0, y=1, xanchor='right', yanchor='top')
    )
    figd.update_xaxes(tickformat=".0%")
    st.plotly_chart(figd)

# Calculate sample size function
def calculate_sample_size(p1, p2, control_share, test_share, alpha, beta):
    if test_share + control_share == 100 :
        ratio = test_share / control_share
        z_score_alpha = norm.ppf(1 - (alpha / 2))
        z_score_beta = norm.ppf(1 - beta)

        p_bar = (p1 + ratio * p2) / (1 + ratio)
        q_bar = 1 - p_bar

        numerator = (z_score_alpha * math.sqrt(p_bar * q_bar * (1 + 1 / ratio))) + (z_score_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio))
        denominator = p1 - p2

        sample_size_control = math.ceil((numerator / denominator) ** 2)
        sample_size_test = math.ceil(sample_size_control * ratio)
        return sample_size_control, sample_size_test
    else:
        raise ValueError('Alarm! test+control must be 100% in total')

if __name__ == "__main__":
    main()
