import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import set_config; set_config(display='diagram')
from sklearn.metrics import mean_absolute_error

from scipy import stats
import xarray as xr
# sklearn preproc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

class Model():

    def __init__(self, file_x='X.nc', file_y='y.nc'):
        self.file_x = file_x
        self.file_y = file_y

    def open_data(self):
        x = xr.open_dataset("X.nc")
        self.X = x.to_array()[0]
        
        Y = xr.open_dataset("Y.nc")
        y = Y.to_dataframe()
        self.y = y['UCUR'].to_numpy()

    def preproc(self):
        self.scaler = StandardScaler()
        return self.scaler
        
    def preproc_fit(self):
        self.scaler.fit(self.X)

    def basic_ml(self, model='ridge'):
        self.pipe_baseline = make_pipeline(self.preproc(), Ridge())
        return self.pipe_baseline
    
    def validation_split(self, test_size=0.3, deep=False):
        if deep:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_preproc, self.y, test_size=test_size, random_state=0)
        else:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=test_size, random_state=0)

        
    def score_baseline(self):
        self.score_base = cross_val_score(self.pipe_baseline, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error').mean()
        print(f"MSE = {-self.score_base}")
        
    def predict_baseline(self):
        self.pipe_baseline.fit(self.X_train,self.y_train)
        self.y_pred_baseline = self.pipe_baseline.predict(self.X)
        print(mean_absolute_error(self.y, self.y_pred_baseline))
        plt.plot(self.y)
        plt.plot(self.y_pred_baseline)

    def preproc_transform(self):
        self.X_preproc = self.scaler.transform(self.X)
    
    def initialize_model(self):

        model = models.Sequential()

        opt = Adam(learning_rate=0.001)

        model.add(layers.Dense(120, activation='relu', input_dim=self.X_train.shape[1]))
        model.add(layers.Dense(80, activation='relu'))
        model.add(layers.Dense(40, activation='relu'))
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(20, activation='relu'))

        model.add(layers.Dense(1, activation='linear')) 

        model.compile(optimizer=opt,
                      loss='msle',
                      metrics='mae')# optimize for the squared log error!

        return model

    def learning_rate(self):
        initial_learning_rate = 0.01

        self.lr = ExponentialDecay(
            initial_learning_rate, decay_steps=500, decay_rate=0.7,
        )

    def define_model(self):
        self.model = self.initialize_model()
        print()
        print(f"learning rate: {0.001}")
        print('Loss = msle')
        print('Metric = mae')
        print('Opt = adam')
        print(self.model.summary())
        
    def model_fit(self):

        es = EarlyStopping(patience=30, restore_best_weights=True)
        print("Early stop with patience 10")
        print("Batch size 32")
        print("Epoch = 100")
        print("Validation Split = 30%")
        
        self.history = self.model.fit(self.X_train, self.y_train,
                            validation_split=0.3,
                            epochs=300,
                            batch_size=16, 
                            verbose=1, 
                            callbacks=[es])

    def plot_history(self):
        fig1, ax = plt.subplots(1,2)
        fig1.set_size_inches(16, 5)
        ax[0].plot(np.sqrt(self.history.history['loss']))
        ax[0].plot(np.sqrt(self.history.history['val_loss']))
        ax[0].set_title('Model Loss')
        ax[0].set_ylabel('MSLE')
        ax[0].set_xlabel('Epoch')
        ax[0].legend(['Train', 'Val'], loc='best')
        ax[1].plot(np.sqrt(self.history.history['mae']))
        ax[1].plot(np.sqrt(self.history.history['val_mae']))
        ax[1].set_title('Model Metric')
        ax[1].set_ylabel('MAE')
        ax[1].set_xlabel('Epoch')
        ax[1].legend(['Train', 'Val'], loc='best')

        plt.show()

    def model_evaluate(self):
        values = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        print(f'MSLE = {values[0]}')
        print(f'MAE = {values[1]}')

    def predict_deep(self):
        self.y_pred = self.model.predict(self.scaler.transform(self.X))
        plt.plot(self.y)
        plt.plot(self.y_pred)
