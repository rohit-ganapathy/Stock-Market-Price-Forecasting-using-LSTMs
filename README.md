# Stock market price forecasts using LSTMs
An lstm based forecasting model that predicts future stock prices after being trained on sequences of prices.
Preprocessing the data involves time-series concepts. I've tried to apply various operations (logging, differencing) in order to make the time series stationary before snipping it into sequences and feeding it to the network.
Tensorboard integrations have been made to visualize loss. 

## Getting Started
 *run.py -> begins training and then inference post training
 *forecasting_model.py-> class contains the architecture of the model as well as functions for inference, plotting and training.  
 *pipeline.py-> data preprocessing pipeline that reads the data into a dataframe, applies various transformations and creates batches

### Dependencies

* python 3+
* numpy
* pandas
* plotly==3.2.1
* tensorflow==1.9.0
* tensorflow-hub==0.1.1
* scikit-learn==0.19.1

