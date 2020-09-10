import pandas as pd
import numpy as np
import pygmo as pg
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from random import randint
from sklearn.linear_model import LinearRegression

import Data_processing
import Turbine_SS_model
import Turbine_dynamic_model


class turbine_problem(object):

    def __init__(self, file_path, weighted_err=False):
        # model param:
        self.param = {
            'weighted_err': weighted_err,
            'dt': 0.1,  # sec
            'pump_level': 0,
            'PT2_constants': [-1.8998, 0, 0.0452],
            'PT1_constants' : [[129.8, -3.373],
                               [125.5, -3.3638],
                               [118.59, -3.2444]],
            'Tw': 1.88,
            'H': 0.266,
            'torque_constant': 46.541,  # mNm/A
            'torque_resistance': 10.506,  # mNm
            'speed_constant': 213,  # rad/s/V
            'internal_resistance': 1.67,
            'V_intercept': 1.3,
            'minor_loss': 0.491,  # psi
            'density_water': 997,  # kg/m^3
            'm3_to_GPM': 15.89806024,
            'psi_to_Pa': 6894.76,
            'RPM_to_radpersec': 0.104719755
        }

        self.param_headers = [
            'Tw',
            'H',
        ]

        self.x_headers = [
            't',
            'PT1',
            'PT2',
            'DP',
            'w',
            'q',
            'g',
            'Tm',
            'Te',
            'I',
            'V'
        ]
        self.error_headers = [
            'w',
            'q',
            'Tm',
            'DP'
        ]

        self.x = {}
        self.prediction = []
        self.error = []
        self.run_time = 0

        self.data = Data_processing.turbine_data(file_path)
        self.SS_model = Turbine_SS_model.SS_model(load_model=True, folder_name="SS_models/")
        self.dynamic_model = Turbine_dynamic_model.dynamic_model(self.param)

        self.linear_regression()

        if weighted_err:
            self.get_weights()

    def update_param(self, p):

        for i, param in enumerate(self.param_headers):
            self.param[param] = p[i]

    def initiate_x(self, data):

        for head in self.x_headers:
            if head == 'Te':
                self.x['Te'] = self.x['Tm']
            else:
                self.x[head] = data[self.data.transient_col.index(head)]

        self.prediction = np.array(
            [list(self.x[head] for head in self.x_headers)])

    def append_x(self):

        self.prediction = np.vstack(
            (self.prediction, list(self.x[head] for head in self.x_headers)))

    def update_x(self, x):

        for head in x.keys():
            self.x[head] = x[head]

    def save_predict(self):

        if len(self.prediction) > 0:

            data_df = pd.DataFrame(self.prediction)
            data_df.columns = self.x_headers
            data_df.to_csv('predict_' + str(self.param['pump_level']) + '.csv')

    def get_weights(self):

        self.weight = [0, 0, 0, 0]
        base_values = [300, 27, 130, 8]
        param_origin = [3.0484, 0.191457]
        factor = 1.2

        for i, data in enumerate(self.data.transient_list_np):
            self.param['pump_level'] = i

            self.update_param(param_origin)
            self.predict(data)
            predict_origin = self.prediction.copy()

            for z in range(len(param_origin)):
                new_param = param_origin.copy()
                new_param[z] = new_param[z]*factor
                self.update_param(new_param)
                self.predict(data)
                predicted_data = self.prediction.copy()
                size = len(predicted_data)
                for j, var in enumerate(self.error_headers):
                    self.weight[j] += np.sum((predict_origin[:,self.x_headers.index(var)] / base_values[j] 
                                            - predicted_data[:,self.x_headers.index(var)] / base_values[j])**2)/size

        total_w = np.sum(self.weight)
        self.weight = self.weight / total_w

        print(self.weight)

    def linear_regression(self):

        y = self.data.steady_state_np[:,self.data.steady_state_col.index('Tm')]
        X = self.data.steady_state_np[:,self.data.steady_state_col.index('I'):self.data.steady_state_col.index('I')+1]

        reg = LinearRegression().fit(X, y)

        self.param['torque_constant'] = reg.coef_[0]
        self.param['torque_resistance'] = reg.intercept_

        print("\nI fitting score: ")
        print(reg.score(X, y))

        y = self.data.steady_state_np[:,self.data.steady_state_col.index('V')]
        X = self.data.steady_state_np[:,self.data.steady_state_col.index('w'):self.data.steady_state_col.index('I')+1]

        reg = LinearRegression().fit(X, y)

        self.param['speed_constant'] = 1/reg.coef_[0]
        self.param['internal_resistance'] = -1*reg.coef_[1]
        self.param['V_intercept'] = -1*reg.intercept_

        print("\nV fitting score: ")
        print(reg.score(X, y))
        print()

    def predict(self, data):

        self.prediction = []
        self.initiate_x(data[0])

        for i in range(1, len(data)):
            self.x['t'] = data[i, 0]
            self.x['g'] = max(data[i, 3], 0.0)
            self.x['I'] = data[i, 4]

            self.param['dt'] = self.x['t'] - self.prediction[-1, 0]

            self.update_x(self.SS_model.predict(
                {'q': self.x['q'], 
                 'g': self.x['g'], 
                 'w': self.x['w']}))
            self.update_x(self.dynamic_model.step(self.x))
            self.append_x()

    def eveluate_error(self, data):

        self.error = []
        for i, obj in enumerate(self.error_headers):

            if self.param['weighted_err']:
                weight = self.weight[i]
            else:
                weight = 1

            self.error.append(weight * np.max((data[:, self.data.transient_col.index(obj)] -
                                    self.prediction[:, self.x_headers.index(obj)])**2))

        return sum(self.error)

    def fitness(self, p, save=False):

        start_loop_time = datetime.now()
        self.update_param(p)

        try:
            error = 0
            for i, data in enumerate(self.data.transient_list_np):
                self.param['pump_level'] = i
                self.predict(data)
                error += self.eveluate_error(data)

                if save:
                    self.save_predict()

        except:
            print("bad param:")
            print(p)
            return [10000000]

        self.run_time = (datetime.now() - start_loop_time).total_seconds()
        # print("run time (ms): " +
        #       str("%.1f" % round(self.run_time*1000, 1)))

        return [error]

    def get_bounds(self):

        lower_bound = [
            1,
            0.1
        ]
        upper_bound = [
            10,
            10
        ]

        return (lower_bound, upper_bound)

class parameter_identification(object):

    def __init__(self, data_file_path, save_file_path, file_name, load=False):

        self.load = load
        self.save_file_path = save_file_path
        self.file_name = file_name
        self.param = {
            'weighted_err': True,
            'sessions': 2,
            'gen': 50,
            'pop_size': 16,
            'eta1': 0.2,
            'eta2': 0.2
        }
        self.logs = []
        self.best = []
        self.pop_list = []

        self.turbine = turbine_problem(data_file_path, weighted_err=self.param['weighted_err'])
        self.problem = pg.problem(self.turbine)

        if self.load:
            self.load_results()

    def run(self):

        self.algo = pg.algorithm(pg.pso(
            gen=self.param['gen'],
            eta1=self.param['eta1'],
            eta2=self.param['eta2']
        ))
        self.algo.set_verbosity(1)

        for i in range(self.param['sessions']):
            
            print("\n"+str(i+1) + " out of " + str(self.param['sessions']) + " started. \n")

            self.algo.set_seed(randint(0,1000))

            self.pop = pg.population(self.problem, size=self.param['pop_size'], seed=randint(0,1000))
            self.pop = self.algo.evolve(self.pop)

            self.process_results()
            self.save_results()

            print(self.pop)
            print("\n" + str(i+1) + " out of " + str(self.param['sessions']) + " saved.")
        
    
    def process_results(self):

        self.pop_list.append(self.pop)

        self.uda = self.algo.extract(pg.pso)
        self.get_logs()
        
        self.best.append(self.pop.get_x()[self.pop.best_idx()])


    def get_param(self, j):

        self.best_param = {}
        for i, head in enumerate(self.turbine.param_headers):
            self.best_param[head] = self.best[j][i]

        print("best param: ")
        print(self.best_param)
        print('\n')

    def get_logs(self):

        logs = []
        for gen in range(len(self.uda.get_log())):
            logs.append(self.uda.get_log()[gen])

        self.logs.append(logs)

    def save_results(self):

        pickle.dump(self.param, open(self.save_file_path + self.file_name + "param.sav", 'wb'))
        pickle.dump(self.pop_list, open(self.save_file_path + self.file_name + "pop_list.sav", 'wb'))
        pickle.dump(self.logs, open(self.save_file_path + self.file_name + "logs.sav", 'wb'))
        pickle.dump(self.best, open(self.save_file_path + self.file_name + "best.sav", 'wb'))

    def load_results(self):

        self.param = pickle.load(open(self.save_file_path + self.file_name + "param.sav", 'rb'))
        self.pop_list = pickle.load(open(self.save_file_path + self.file_name + "pop_list.sav", 'rb'))
        self.logs = pickle.load(open(self.save_file_path + self.file_name + "logs.sav", 'rb'))
        self.best = pickle.load(open(self.save_file_path + self.file_name + "best.sav", 'rb'))

        self.turbine.fitness(self.best[0], save = True)

        param_headers = [
            'Tw',
            'H',
            'torque_constant',
            'torque_resistance', 
            'speed_constant',  
            'internal_resistance',
            'V_intercept'
        ]

        print('best param saved: ')
        best_param = {}
        for p in param_headers:
            best_param[p] = self.turbine.param[p]
            print(p)
            print(best_param[p])

        pickle.dump(best_param, open(self.save_file_path + self.file_name + "_best_param.sav", 'wb'))

        self.param['weighted_err'] = False

        print("\nmax predictions error:")
        error = np.array([0, 0, 0, 0])
        for i, data in enumerate(self.turbine.data.transient_list_np):
            self.turbine.param['pump_level'] = i
            self.turbine.predict(data)
            temp = self.turbine.eveluate_error(data)
            error = np.maximum(error, np.sqrt(np.array(self.turbine.error)))

        for i, head in enumerate(self.turbine.error_headers):
            print(head)
            print(error[i])


if __name__ == '__main__':

    import Parameter_identification

    data_file_path = "/home/lilly/Energy_Harvester/Processed_data2"
    save_file_path = "/home/lilly/Energy_Harvester/scripts/Parameter_identification"
    file_name = '/2020_02_21_2param'

    optimizer = parameter_identification(data_file_path, save_file_path, file_name, load=True)
    # optimizer.run()
