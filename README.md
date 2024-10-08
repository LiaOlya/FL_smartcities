# Federated Learning for Smart Cities

In this project we forecast short-term residential energy consumption using "SmartMeter Energy Consumption Data" provided by London Datastore. Our goal is to demonstrate the advantages of Federated Learning compared to centralized methods. We trained our models using both centralized and federated approaches to highlight the benefits of federated learning using centralized setting as a benchmark. In our project, the open source framework Flower was used to federate model training and evaluation. 

We started our project by preprocessing the originial dataset making it ready for model training. Specifically, we utilized two different preprocessing approaches: In one approach, we excluded the column is_weekend to enhance computational efficiency. In the other approach, we retained this column. Afterwards, we used two LSTM models to forecast hourly energy consumption. The models differ in the number of LSTM layers that they have. Stacked LSTM model consists of four LSTM layers while simple LSTM model has only one LSTM layer. 

The code for preprocessing the original dataset as well as federated / centralized model training scenarios are available in this repository.
