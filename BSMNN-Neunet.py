import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import keras 
import keras.backend as K
import tensorflow

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from py_vollib import black_scholes_merton as bsm
from progressbar import ProgressBar
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import uniform
from keras.optimizers import Adagrad
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import LearningRateScheduler



path = 'file:///C:/Users/yezda/Downloads/dataFullV2.csv'
df_training = pd.read_csv(path)


inputs = df_training[['K','S','q','r','sigma','t']]
outputs = df_training[['price']]

# Tried to see if scaling data would improve accuracy, no noticeable difference.
'''scaler = StandardScaler()

# Fit the scaler to your DataFrame and transform it
inputs_scaled = pd.DataFrame(scaler.fit_transform(inputs), columns=inputs.columns)
outputs_scaled = pd.DataFrame(scaler.fit_transform(outputs), columns=outputs.columns)
'''

x_train,x_test,y_train,y_test = train_test_split(inputs, outputs, test_size=0.1, random_state=7)

def baseline_model():
    # create model
    i = Input(shape=(6,))
    x = Dense(10, activation='relu')(i)
    y = Dense(10, activation='relu')(x)
    o = Dense(1)(y)
    model = Model(i, o)
    # Tried using SGD optimizer with specified learning rate and momentum, minimal improvement in accuracy.
    '''optimizer = Adagrad(learning_rate=0.001,initial_accumulator_value=0.1,epsilon=1e-06,weight_decay=None,clipnorm=None,clipvalue=None,global_clipnorm=None,use_ema=False,ema_momentum=0.99,ema_overwrite_frequency=None,jit_compile=True,name="Adagrad")'''

    model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])
    return model
model_full = baseline_model()
# Tried using a learning rate scheduler, again: some but not much better.
'''
def learning_rate_schedule(epoch):
    initial_lr = 0.001  # Initial learning rate
    decay = 0.95  # Learning rate decay factor
    return initial_lr * (decay ** epoch)

# Create the LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(learning_rate_schedule)
'''
percentage_differences = []
history_full = model_full.fit(x_train, y_train, epochs=50, batch_size=64)
test_output = model_full.predict(x_test)

print('______________________________________EVALUATION METRICS______________________________________')
print('______________________________________________________________________________________________')
print (f'mean squared error for option pricing model on out of sample test is {mean_squared_error(y_test,test_output)}')


