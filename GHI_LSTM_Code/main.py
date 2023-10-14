from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import torch
from torch import nn
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from dataset import SequenceDataset
from torch.utils.data import DataLoader

from model import *

st.title("Solar Prediction Application")
# Design tab
tab1, tab2 = st.tabs(["🗃 Data","📈 Chart"])

# Create button for uploading data
csv_file = st.sidebar.file_uploader(
    label="Upload data in here",
    type=["csv"]
)

if csv_file is None:
    st.subheader("# Please upload data in left corner")
else:
    solargis = pd.read_csv(csv_file, comment="#", sep=";")
    # Draw Original Data Table
    tab1.header("# Original Data")
    tab1.write(solargis)

    # Merge "DateTime" = "Date Time"
    solargis["DateTime"] = solargis["Date"] + " " + solargis["Time"]
    solargis = solargis.set_index("DateTime")

    # Choose the range of the data from 2016-2019
    solargis_2016_to_2019 = solargis.loc["01.01.2016 00:30":"31.12.2019 23:30"]

    # Design which are selected input 
    st.sidebar.write("Input Variables")
    AP = st.sidebar.checkbox("Atmospheric pressure [hPa]", value=True)
    RH = st.sidebar.checkbox("Relative humidity [%]", value=True)
    TEMP = st.sidebar.checkbox("Air temperature at 2m [deg.C]", value=True)
    SE = st.sidebar.checkbox("Sun altitude (elevation) angle [deg.]", value=True)
    SA = st.sidebar.checkbox("Sun azimuth angle [deg.]", value=True)
    WS = st.sidebar.checkbox("Wind speed at 10m [m/s]", value=True)
    WD = st.sidebar.checkbox("Wind direction at 10m [deg.]", value=True)
    PWAT = st.sidebar.checkbox("Precipitable water [kg/m2]", value=True)

    input_variables = []
    if AP:
        input_variables.append("AP")
    if RH:
        input_variables.append("RH")
    if TEMP:
        input_variables.append("TEMP")
    if SE:
        input_variables.append("SE")
    if SA:
        input_variables.append("SA")
    if WS:
        input_variables.append("WS")
    if WD:
        input_variables.append("WD")
    if PWAT:
        input_variables.append("PWAT")

    # Choose the selected output 
    st.sidebar.write("Ouput Variables")
    output_variables = st.sidebar.selectbox(
        'What is output do you need to predict ?',
        ("GHI", "DNI", "DIF", "GTI"))     

    # Draw Training Data Table which contain all needed data
    data = solargis_2016_to_2019[input_variables + ["GHI"]]
    tab1.header("# Training data")
    tab1.write(data)

    # target_name = output_variables
    # Now always to use LSTM model
    target_name = "GHI"
    features = list(data.columns.difference([target_name]))

    # Choose the model for predicting
    st.sidebar.write("Type of model")
    model_type = st.sidebar.selectbox(
        'What type of model do you need to predict ?',
        ("LSTM", "RNN", "ANN"))      

    # Design a slider for choosing which is time forecasting 
    forecast_lead = st.sidebar.slider(
        label="How far ahead do you want to forecast? (hour)", 
        value=1,
        min_value=1,
        max_value=24,
        step=1
    )

    target = f"{target_name}_lead{forecast_lead}hour"
    data[target] = data[target_name].shift(-forecast_lead)
    data = data.iloc[:-forecast_lead]

    # Design a slider for choosing which is % of predict size = predict size / total size
    test_size = st.sidebar.slider(
        label="Choose the value predicting size ? (%)",
        value=20,
        min_value=0,
        max_value=100,
        step=5
    )

    # Set train size & predict size
    test_size = test_size / 100
    test_start = int(data.shape[0] * (1 - test_size))

    data_train = data.iloc[:test_start]
    data_test = data.iloc[test_start:]

    # Normalize Data
    mean = data_train.mean()
    std = data_train.std()
    data_train = (data_train - mean) / std
    data_test  = (data_test - mean) / std

    batch_size = 4
    sequence_length = 30

    train_dataset = SequenceDataset(
        data_train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        data_test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    learning_rate = 5e-5
    num_hidden_units = 16

    # Need check model_type = LSTM or RNN or ANN 
    # Now always to use LSTM model
    model = ShallowRegressionLSTM(num_sensors=len(features), hidden_units=num_hidden_units)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Button Start Training/Reset
    tab1.divider()
    tab1.header("# Please choose next action")
    tab1.button("Reset", type="primary")
    if tab1.button("Start training"):
        tab1.header("# Progress:")
        train_progess_bar = tab1.progress(0)
        progress = 0
        num_epoches = 5

        train_avg_losses = [np.nan]*num_epoches
        test_avg_losses = [np.nan]*num_epoches

        tab1.header("# Losses:")
        fig, ax = plt.subplots()
        ax.set_xlim(0, num_epoches)
        ax.set_ylim(0, 1)
        line_train, = ax.plot(np.arange(num_epoches), train_avg_losses, label="train")
        line_test,  = ax.plot(np.arange(num_epoches), test_avg_losses, label="test")
        the_plot = tab1.pyplot(plt)

        for ix_epoch in range(num_epoches):
            train_avg_loss = train_model(train_loader, model, loss_function, optimizer=optimizer)
            test_avg_loss = test_model(test_loader, model, loss_function)

            progress += 1/num_epoches
            train_progess_bar.progress(progress)

            train_avg_losses[ix_epoch] = train_avg_loss
            test_avg_losses[ix_epoch]  = test_avg_loss

            line_train.set_ydata(train_avg_losses)
            line_test.set_ydata(test_avg_loss)
            the_plot.pyplot(plt)

        tab1.success('Training finished', icon="✅")
        tab1.caption("Please change to Chart tab to see the result")
        tab1.caption("Please press Reset button to reset training")
        tab1.caption("Please press Browser files button to choose another source data")        
        train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        ystar_col = "Model forecast"
        data_train[ystar_col] = predict(train_eval_loader, model).numpy()
        data_test[ystar_col] = predict(test_loader, model).numpy()

        df_out = pd.concat((data_train, data_test))[[target, ystar_col]]

        for c in df_out.columns:
            df_out[c] = df_out[c] * std[target] + mean[target]

        
        pio.templates.default = "plotly_white"

        plot_template = dict(
            layout=go.Layout({
                "font_size": 12,
                "xaxis_title_font_size": 24,
                "yaxis_title_font_size": 24})
        )

        fig = px.line(df_out, labels=dict(created_at="Time axis", value="W/m2"))
        fig.add_vline(x=test_start, line_width=4, line_dash="dash")
        fig.add_annotation(xref="paper", x=1, yref="paper", y=1, text="Test set start", showarrow=False)
        fig.update_layout(
            template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
        )
        tab2.plotly_chart(fig)