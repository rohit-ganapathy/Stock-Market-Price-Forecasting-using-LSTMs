import numpy as np
import os
import random
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from pipeline import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt



class configuration():
    
 
    max_epoch = 500
    init_learning_rate = 0.01
    learning_rate_decay = 0.99
    init_epoch = 1
    batch_size = 64
    keep_prob = 0.9
    input_size  = 1
    output_length=1





config=configuration




class LstmRNN(object):
    def __init__(self, sess,
                 lstm_size=128,
                 num_layers=2,
                 num_steps=30,
                 input_size=1,
                 logs_dir="logs"):
        """
        Construct a RNN model using LSTM cell.
        Args:
            sess:
            lstm_size (int)
            num_layers (int): num. of LSTM cell layers.
            num_steps (int)
            input_size (int)
            keep_prob (int): (1.0 - dropout rate.) for a LSTM cell.
            checkpoint_dir (str)
        """
        self.sess = sess
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size
        self.logs_dir = logs_dir
        self.pipeline=pipeline(pp="dls",input_length=self.num_steps, output_length=config.output_length, lag=1, batch_size=config.batch_size)
        self.test_data_iterator=self.pipeline.GenerateEpoch()
        self.build_graph()
        



    def build_graph(self):
        
        tf.reset_default_graph()
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")
        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
        ) if self.num_layers > 1 else _create_one_cell()

        
        # Run dynamic RNN
        val, state_ = tf.nn.dynamic_rnn(cell,self.inputs, dtype=tf.float32, scope="dynamic_rnn")

        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        val = tf.transpose(val, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
        ws = tf.Variable(tf.truncated_normal([self.lstm_size, self.input_size]), name="w")
        bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")
        self.pred = tf.matmul(last, ws) + bias

        self.last_sum = tf.summary.histogram("lstm_state", last)
        self.w_sum = tf.summary.histogram("w", ws)
        self.b_sum = tf.summary.histogram("b", bias)
        self.pred_summ = tf.summary.histogram("prediction", self.pred)

        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")

        # Separated from train loss.
        self.loss_test = tf.placeholder(tf.float32, None, name="test_loss")

        self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
        self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test,collections=['per-epoch'])
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)
        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    def train(self):
      
        # Set up the logs folder
        self.merged_summary = tf.summary.merge_all()
        self.per_epoch = tf.summary.merge_all(key='per-epoch')
        self.writer = tf.summary.FileWriter(os.path.join("./logs", "model"))
        self.writer.add_graph(self.sess.graph)
        
        tf.global_variables_initializer().run()
        random.seed(time.time())


        for epoch in range(config.max_epoch):
            
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )
            iteration=0
            for batch in self.test_data_iterator:
                
                iteration+=1
                batch_x=batch[0]
                batch_y=batch[1]
                
                train_data_feed = {
                        self.learning_rate: learning_rate,
                        self.keep_prob: config.keep_prob,
                        self.inputs: batch_x,
                        self.targets: batch_y,
                    }
                train_loss,_,summary = self.sess.run(
                        [self.loss, self.optim,self.merged_summary], train_data_feed)
                
                
                print("epoch:",epoch,"iteration:",iteration,"loss:",train_loss)
        
            self.y_predicted=self.Inference()       
            test_data_feed = { self.loss_test:self.RMSE()}
                        
                        
                    
            
            test_loss,summary_test=self.sess.run([self.loss_test,self.per_epoch],test_data_feed)
            print("test-loss:",test_loss)
            self.writer.add_summary(summary, global_step=epoch)
            self.writer.add_summary(summary_test, global_step=epoch)
            
            
            if epoch%20==0:
                self.pipeline.Plot(self.y_predicted)
                
                
                
            self.test_data_iterator=self.pipeline.GenerateEpoch() 
        
    def Inference(self):
        
        self.infer=self.pipeline.train_seq[-(self.num_steps):]
        self.predicted_series=[]
        for i in range(len(self.pipeline.actual)):
             self.feed=np.expand_dims(self.infer,1)
             self.feed=np.expand_dims(self.feed,0)
             self.prediction= self.sess.run(self.pred,feed_dict={self.inputs:self.feed,self.keep_prob:1})
             self.prediction=np.asscalar(self.prediction)
             self.predicted_series.append(self.prediction)
             self.infer.pop(0)
             self.infer.append(self.prediction)
            
        
        return self.predicted_series
        
    def RMSE(self):
        
        actual=self.pipeline.seq
        self.final_predicted_series=self.pipeline.Reverse(self.y_predicted)
        rmse = sqrt(mean_squared_error(actual, self.final_predicted_series))
        
        return rmse
        
        
        
        