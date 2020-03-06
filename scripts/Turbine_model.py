
import glob
import os

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#base class that read, clean, and store all experimental data
class turbine_data(object):

    def __init__(self, file_path):

        #constants
        self.RPM_to_radpersec = 0.104719755
        self.minor_loss = 0.491 #psi
        self.density_water = 997 #kg/m^3
        self.m3_to_GPM = 15.89806024
        self.psi_to_Pa = 6894.76

        self.all_files = glob.glob(os.path.join(file_path, "*.csv")) #make list of paths
        self.file_name = []
        self.steady_state_df = []
        self.transient_df = []
        self.limits = []

        #steady state data
        self.steady_state_headers = ['Speed (rad/s)','GV (deg)','Flow Rate (GPM)','DP (psi)','torque (mNm)','turbine eff']
        self.steady_state_np = []
        self.steady_state_len = 0

        #transient data
        self.transient_headers = ['Time (sec)', 'GV (deg)', 'I (A)', 'Speed (rad/s)', 'Flow Rate (GPM)', 'DP (psi)','torque (mNm)', 'V (V)' ]
        self.transient_list_np = []
        self.transient_list_len = []

        self.read_all()
        self.clean_steady_state_data()
        self.clean_transient_data()
        
    def read_all(self):

        print("\nloading files: ")
        ss_file_loaded = False

        for f in self.all_files:

            file_name = os.path.splitext(os.path.basename(f))[0]  # Getting the file name without extension
            self.file_name.append(file_name)

            dataframe = pd.read_csv(f)

            dataframe['Speed (rad/s)'] = dataframe['Speed (RPM)']*self.RPM_to_radpersec
            dataframe['turbine power (w)'] = dataframe['Speed (rad/s)'] * dataframe['torque (mNm)'] / 1000
            dataframe['fluid power (w)'] = dataframe['Flow Rate (GPM)'] / self.m3_to_GPM * (dataframe['DP (psi)']\
                                           - self.minor_loss)*self.psi_to_Pa/self.density_water
            dataframe['turbine eff'] = dataframe['turbine power (w)'] / dataframe['fluid power (w)']

            if "SS" in file_name:
                if ss_file_loaded:
                    self.steady_state_df = pd.concat([self.steady_state_df, dataframe])
                else:
                    ss_file_loaded = True
                    self.steady_state_df = dataframe
            else:
                self.transient_df.append(dataframe)

            print(file_name)

    def clean_steady_state_data(self):

        self.steady_state_np = np.array(self.steady_state_df[self.steady_state_headers].sort_values('GV (deg)').values)
        self.steady_state_len  = len(self.steady_state_np)
        print("\ntotal num of SS data pts: " +str(self.steady_state_len))

        np.random.shuffle(self.steady_state_np)

    def clean_transient_data(self):

        total_data = 0

        for df in self.transient_df:
            self.transient_list_np.append(np.array(df[self.transient_headers].values))
            self.transient_list_len.append(len(self.transient_list_np[-1]))
            total_data += self.transient_list_len[-1]

        print("\ntotal num of transient data pts: " +str(total_data))

    def get_limits(self):

        self.limits = []

        GVs = self.steady_state_df['GV (deg)'].unique()
        for GV_angle in GVs:
            temp = [GV_angle]
            temp.append(self.data_df[(self.data_df['GV (deg)'] == GV_angle)]['Speed (rad/s)'].min())
            temp.append(self.data_df[(self.data_df['GV (deg)'] == GV_angle)]['Speed (rad/s)'].max())
            self.limits.append(temp)
        
        return self.limits
        
class SS_model_base(object):

    def __init__(self, data, x_headers, y_header):

        self.y_header = y_header
        self.x_headers = x_headers

        self.X = np.empty([data.steady_state_len, len(x_headers)])
        self.y = np.empty([data.steady_state_len, 1])
        self.len = data.steady_state_len

        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.X_pred = []
        self.y_pred = []
        self.y_std = []

        self.set_xy_data(data)

    def set_xy_data(self, data):

        for i, head in enumerate(self.x_headers):
            self.X[:, i] = data.steady_state_np[:,data.steady_state_headers.index(head)]

        self.y[:,0] = data.steady_state_np[:,data.steady_state_headers.index(self.y_header)]

    def get_train_error(self):

        self.X_pred = self.X_train
        self.model_predict()

        return np.sum((self.y_pred-self.y_train)**2)

    def get_test_error(self):
        
        self.X_pred = self.X_test
        self.model_predict()

        return np.sum((self.y_pred-self.y_test)**2)

    def predict(self, X):

        if isinstance(X[0], float):
            X = [X]

        self.X_pred = X
        self.model_predict()

    def train(self):

        self.X_train = self.X
        self.y_train = self.y

        self.fit()

        return self.get_train_error()

    def train_with_cross_validation(self, split = 9):

        test_error = []
        train_error = []

        for j in range(split):

            start = j*self.len//split
            if j == split - 1:
                end = self.len
            else:
                end = (j+1)*self.len//split

            self.X_train, self.X_test = np.concatenate([self.X[0:start,:], self.X[end:,:]]), self.X[start:end,:]
            self.y_train, self.y_test = np.concatenate([self.y[0:start,:], self.y[end:,:]]), self.y[start:end,:]

            self.fit()

            test_error.append(self.get_test_error())
            train_error.append(self.get_train_error())

        return [sum(test_error), sum(train_error)/(split-1)]


class SS_model_GP(SS_model_base):

    def __init__(self, data, x_headers, y_header, scale = 30**2, RBF_scale = [2.91, 174, 12.8], noise = 0.617):

        super().__init__(data, x_headers, y_header)

        self.scale = scale
        self.RBF_scale = RBF_scale
        self.noise = noise

        self.kernel = self.scale * RBF(length_scale=self.RBF_scale, length_scale_bounds=(1e-3, 100000.0))\
                                    + WhiteKernel(noise_level=self.noise, noise_level_bounds=(1e-10, 1e+10))

        self.model = GaussianProcessRegressor(kernel=self.kernel)

    def update_hyperparameters(self, param):

        self.scale = param['param1']
        self.RBF_scale = [param['param2'], param['param3'], param['param4']]
        self.noise = param['param5']

        self.kernel = self.scale * RBF(length_scale=self.RBF_scale, length_scale_bounds=(1e-3, 100000.0))\
                                    + WhiteKernel(noise_level=self.noise, noise_level_bounds=(1e-10, 1e+10))

        self.model.set_params(kernel = self.kernel)
            
    def get_hyperparameters(self):

        return self.model.kernel_                  

    def model_predict(self):

        self.y_pred, self.y_std = self.model.predict(self.X_pred, return_std=True)

    def fit(self):

        self.model.fit(self.X_train, self.y_train)

    def grid_search(self):

        scale = [50**2, 100**2, 300**2]
        RBF_scale = [
            [0.1, 1, 5],
            [100, 300],
            [1, 10, 50]
        ]
        noise = [1e-1, 1]

        param_grid = {'param1': scale, 'param2' : RBF_scale[0], 'param3':RBF_scale[1], 'param4':RBF_scale[2], 'param5':noise}
        grid = ParameterGrid(param_grid)

        best_settings =[]
        best_test_error = 99999
        num_grid_search = len(grid)

        for i, param in enumerate(grid):
        
            self.update_hyperparameters(param)

            test_error, train_error= self.train_with_cross_validation()

            if test_error < best_test_error:
                best_test_error = test_error
                best_settings = param

            print("\nsearching: " + str(i) + " / " +str(num_grid_search))
            print('params: ')
            print(param)
            print(self.get_hyperparameters())
            print("train_error: " + str(train_error))
            print("test_error: " + str(test_error))

        self.update_hyperparameters(best_settings)

class SS_model(object):

    def __init__(self, input_var, output_var, data = None):

        self.models = []
        self.path = "SS_models/"
        self.x_headers = input_var
        self.y_headers = output_var

        print("\ninputs: "+ '['+' | '.join(input_var) +']')
        print("outputs: "+ '['+' | '.join(output_var)+']')

        if data == None:
            self.load_models()
        else:
            for y in output_var:
                self.models.append(SS_model_GP(data, input_var, y))

    def train(self):

        for model in self.models:
            train_error = model.train()

            print("\nGP model: " + model.y_header)
            print(model.get_hyperparameters())
            print("training error: " + str(train_error))

    def predict(self, X):

        pred = np.empty([len(X), len(self.models)])
        for i, model in enumerate(self.models):
            model.predict(X)
            pred[:,i] = model.y_pred[:,0]

        for i, x in enumerate(X):
            print("\ninputs: "+ '['+' '.join(str(j) for j in x) +']')
            print("outputs: "+ '['+' '.join(str(j) for j in pred[i,:])+']')

    def train_with_cross_validation(self):

        for model in self.models:
            test_error, train_error= model.train_with_cross_validation()

            print("\nGP model: " + model.y_header)
            print(model.get_hyperparameters())
            print("training error: " + str(train_error))
            print("testing error: " + str(test_error))

    def grid_search(self):

        for i, model in enumerate(self.models):
            if i == 1:
                model.grid_search()

    def save_models(self):

        for i, model in enumerate(self.models):
            if i == 1:
                filename = "model_" + str(i) + ".sav"
                pickle.dump(model, open(self.path + filename, 'wb'))

    def load_models(self):

        self.models = []
        all_files = glob.glob(os.path.join(self.path, "*.sav")) #make list of paths

        print("\nloading models: ")
        for f in all_files:
            file_name = os.path.splitext(os.path.basename(f))[0]  # Getting the file name without extension
            self.models.append(pickle.load(open(self.path + file_name + ".sav", 'rb')))
            print(file_name)
        print()





if __name__ == '__main__':
    
    file_path=r"C:\Users\lilly\OneDrive\Documents\1.0_Graduate_Studies\5.0 Energy havester\5.8_code\Energy_Harvester\Processed_data2"

    data = turbine_data(file_path)

    #Steady State Model
    output_var = ['GV (deg)', 'torque (mNm)']
    input_var = ['DP (psi)', 'Speed (rad/s)', 'Flow Rate (GPM)' ]

    steady_state_model = SS_model(input_var, output_var, data=None)
    # steady_state_model.grid_search()
    #steady_state_model.train()
    # steady_state_model.train_with_cross_validation()
    # steady_state_model.save_models()
    steady_state_model.predict([[ 19.9, 567, 25.0],
                                [ 19.2, 523, 25.0],
                                [ 7.3, 412, 27.3]])