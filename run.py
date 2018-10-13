import numpy as np
import forecasting_model 
import tensorflow as tf


model=forecasting_model.LstmRNN(sess="sess")

with tf.Session() as model.sess:
 model.train()
 predicted=model.Inference()
 
 

 
 