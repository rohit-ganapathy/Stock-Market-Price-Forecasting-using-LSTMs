import numpy as np
import pandas as pd
import math
import random
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
plotly.tools.set_credentials_file(username='rohit.ganapathy', api_key='GCX6gT47JoEjc8IkasBP')

"""
class attributes:
 
(type of preprocessing, input seq length, 
output seq length, lag between sequences, batch-size)
                    
types of pre-processing:
    
2. differenced and linearly scaled(dls)
3.log differenced and linearly scaled(ldls)

"""

class pipeline(object):
    
 def __init__(self,
                 pp="dls",
                 input_length=30,
                 output_length=1,
                 lag=1,
                 batch_size=32
                 ):
     
     self.pp = pp
     self.input_length = input_length
     self.output_length = output_length
     self.lag = lag
     self.batch_size=batch_size
     self.raw_df=pd.read_csv("stock-data.csv")
     

     self.seq=list(self.raw_df["Close"])
     self.shape=np.shape(self.seq)
     l=int(0.8*len(self.seq))
     self.prediction_steps=len(self.seq)-l
     
     if self.pp=="dls":
        
        self.diff=[self.seq[i]-self.seq[i-1] for i in range (len(self.seq)) if i>0]
        self.mean=sum(self.diff[:l])/len(self.diff[:l])
        self.max_min=max(self.diff[:l])-min(self.diff[:l])
        self.normalized=[(i-self.mean)/(self.max_min) for i in self.diff]
        self.first=self.seq[0]
            
     if self.pp=="ldls":
                          
        self.log=[math.log(i) for i in self.seq]
        self.first=self.log[0]
        self.diff=[self.log[i]-self.log[i-1] for i in range (len(self.log)) if i>0]
        self.mean=sum(self.diff[:l])/len(self.diff[:l])
        self.max_min=max(self.diff[:l])-min(self.diff[:l])
        self.normalized=[(i-self.mean)/(self.max_min) for i in self.diff]
     
     self.train_seq=self.normalized[:l]   
     self.actual=self.normalized[l:]      
     self.train_x,self.train_y,self.test_x,self.test_y = self.PrepareData(self.normalized)
     self.no_batches=int(len(self.train_x)/self.batch_size)
        
 def PrepareData(self,seq):
     
     
     x,y=[],[]
     
     for i in range(0,len(seq),self.lag):
         
         inp=seq[i:i+self.input_length]
         inp=np.expand_dims(inp,axis=1)
         x.append(inp)
         y.append(seq[i+self.input_length:i+self.input_length+self.output_length])
         if (len(seq)-1)-i<(self.input_length+self.output_length):
             break
         
     l1=int(0.8*len(x))
     
     x_train,y_train=x[0:l1],y[0:l1] 
     x_test,y_test=x[l1:],y[l1:]   
    
        
     return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
 def GenerateEpoch(self):
     
     batch_indices=[i for i in range(self.no_batches)]
     random.shuffle(batch_indices)
     for j in batch_indices:
            batch_X = self.train_x[j * self.batch_size: (j + 1) *self.batch_size]
            batch_Y = self.train_y[j * self.batch_size: (j + 1) * self.batch_size]
            yield batch_X, batch_Y
 
    
 def Reverse(self,predicted):
     
     
     
     self.pred_series=self.train_seq+predicted
     self.inter=[((i*self.max_min)+self.mean) for i in self.pred_series]
     self.reverse=[self.first]
     factor=self.first
     
     for i in self.inter:
            self.reverse.append(factor+i)
            factor=factor+i
            
     
    
     if self.pp=="dls":
        
         return self.reverse
        
               
            
     if self.pp=="ldls":
                          
        return [math.exp(i) for i in self.reverse]
     
     
 def Plot(self,predicted):
     
     
     series_1=self.Reverse(predicted) 

     trace0 = go.Scatter(
     x =self.raw_df["Date"],
     y = series_1,
     name = 'predicted',
     line = dict(
        color = ('rgb(205, 12, 24)'),
        width = 4)
          )
     
     trace1 = go.Scatter(
     x = self.raw_df["Date"],
     y = self.seq,
     name = 'actual',
     line = dict(
        color = ('rgb(22, 96, 167)'),
        width = 4,))
    
    
     data= [trace0, trace1]
            
     layout = dict(title = 'Stock Prices',
              xaxis = dict(title = 'Date'),
              yaxis = dict(title = 'Closing Price'),
              )

     fig = dict(data=data, layout=layout)
     py.iplot(fig, filename='styled-line')
    
    
     
     
     
     
     
     
     
     