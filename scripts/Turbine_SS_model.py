
import glob
import os

import pandas as pd
import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from smt.surrogate_models import KRG, QP

from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        
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

        return  self.y_pred, self.y_std

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

        # self.RBF_scale = [[0.885, 169, 1.25],
        #                   [2.91, 174, 12.8]]

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

class SS_model_KRG(SS_model_base):

    def __init__(self, data, x_headers, y_header, RBF_scale = [2.91, 174, 12.8]):

        super().__init__(data, x_headers, y_header)

        self.RBF_scale = RBF_scale
        self.model = KRG(theta0=RBF_scale, poly = 'quadratic', print_training=False,print_global=False)  

    def update_hyperparameters(self, param):

        self.RBF_scale = param
        self.model = KRG(theta0=param, poly = 'quadratic', print_training=False,print_global=False)  

    def get_hyperparameters(self):

        return self.RBF_scale

    def fit(self):

        self.model.set_training_values(self.X_train, self.y_train)
        self.model.train()

    def model_predict(self):

        X = np.array(self.X_pred)
        self.y_pred = self.model.predict_values(X)
        self.y_std = self.model.predict_variances(X)

class SS_model_QP(SS_model_base):

    def __init__(self, data, x_headers, y_header):

        super().__init__(data, x_headers, y_header)

        self.model = QP(print_training=False,print_global=False)  

    def get_hyperparameters(self):

        return "quadratic model"

    def fit(self):

        self.model.set_training_values(self.X_train, self.y_train)
        self.model.train()

    def model_predict(self):

        X = np.array(self.X_pred)
        self.y_pred = self.model.predict_values(X)
        self.y_std = None


class SS_model(object):

    def __init__(self, data = None, load_model = True, model = 'KRG'):

        self.models = []
        self.models_dict = {}
        self.model_type = model
        self.data = data
        self.path = "SS_models/"

        self.variables = ['DP (psi)', 'Speed (rad/s)', 'Flow Rate (GPM)', 'GV (deg)', 'torque (mNm)']
        self.inputs = list(itertools.combinations(self.variables,3))
        self.outputs = list(itertools.combinations(self.variables,2))[::-1]

        if load_model:
            self.load_models()
        else:
            self.init()

        #model constants:
        self.RPM_to_radpersec = 0.104719755

        #current state:
        self.x = {}
        self.y = {}

    def clear(self):
        self.x = {}
        self.y = {}

    def init(self):

        for i, inp in enumerate(self.inputs):
            self.models_dict[inp] = {}
            for out in self.outputs[i]:
                if self.model_type == 'GP':
                    self.models_dict[inp][out] = SS_model_GP(self.data, inp, out)
                    self.models.append(self.models_dict[inp][out])
                elif self.model_type == 'KRG':
                    self.models_dict[inp][out] = SS_model_KRG(self.data, inp, out)
                    self.models.append(self.models_dict[inp][out])
                elif self.model_type == 'QP':
                    self.models_dict[inp][out] = SS_model_QP(self.data, inp, out)
                    self.models.append(self.models_dict[inp][out])

    def train(self):

        for model in self.models:
            train_error = model.train()

            print("\nModel output: " + model.y_header)
            print("Model inputs: ")
            print(model.x_headers)
            print("training error: " + str(train_error))

    def predict(self, X = {'DP (psi)':0.0, 'Speed (rad/s)':0.0, 'Flow Rate (GPM)':0.0, 'GV (deg)':0.0, 'torque (mNm)':0.0}, 
                      prt_to_screen = False):
        
        model_x_head = []
        model_y_head = []
        model_x = []
        
        self.clear()
        for var in self.variables:
            if var in X.keys():
                self.x[var] = X[var]
                model_x_head.append(var)
                model_x.append(X[var])
            else:
                model_y_head.append(var)

        for y in model_y_head:
            self.y[y] = self.models_dict[tuple(model_x_head)][y].predict(model_x)[0][0][0]

        if prt_to_screen:
            print()
            for var in self.variables:
                if var in self.x.keys():
                    print(var+": "+ str(self.x[var]))
                else:
                    print(var+": "+ str(self.y[var]))

        return self.y

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
        ascii_lowercase = 'abcdefghijklmnopqrstuvwxyz'
        for i, model in enumerate(self.models):
            filename = "model_" + ascii_lowercase[i] + ".sav"
            pickle.dump(model, open(self.path + filename, 'wb'))

    def load_models(self):

        self.models = []
        all_files = glob.glob(os.path.join(self.path, "*.sav")) #make list of paths

        print("\nloading models: ")
        file_name = []
        for f in all_files:
            file_name.append(os.path.splitext(os.path.basename(f))[0])  # Getting the file name without extension

        file_name.sort()

        for f in file_name:
            self.models.append(pickle.load(open(self.path + f + ".sav", 'rb')))
            print(f)
        print()

        for i, inp in enumerate(self.inputs):
            self.models_dict[inp] = {}
            for y, out in enumerate(self.outputs[i]):
                self.models_dict[inp][out] = self.models[i*2 + y]

    def plot_surface(self, x_headers = ['Flow Rate (GPM)', 'GV (deg)', 'torque (mNm)'], 
                           y_headers = ['DP (psi)', 'Speed (rad/s)'],
                           grid_x = 40, grid_y = 40, flow=25.0):

        limits = self.data.get_limits(x_headers)

        X = np.linspace(limits[0][0], limits[0][1], num = grid_x)
        Y = np.linspace(limits[1][0], limits[1][1], num = grid_y)
        X, Y = np.meshgrid(X, Y)

        Z = []
        for model in y_headers:
            temp = np.zeros([grid_x, grid_y])
            for x in range(grid_x):
                for y in range(grid_y):
                    temp[x,y] = self.models_dict[tuple(x_headers)][model].predict([X[x,y],Y[x,y],flow])[0][0][0]
            Z.append(temp)

        fig = plt.figure(figsize=(14,6), tight_layout=True)

        for i, model in enumerate(y_headers):
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            ax.plot_wireframe(X, Y, Z[i], cmap="YlGnBu_r")
            ax.plot_surface(X, Y, Z[i], alpha=0.5, antialiased=True,cmap="YlGnBu_r")

            ax.set_xlabel(x_headers[0], fontsize= 14, labelpad=10)
            ax.set_ylabel(x_headers[1], fontsize= 14, labelpad=10)
            ax.set_zlabel(model, fontsize= 14,  labelpad=5.5, rotation=300)

        plt.autoscale()
        plt.show()


