import pandas as pd
import numpy as np
import pickle
from datetime import datetime

import Data_processing
import Turbine_SS_model
import Turbine_controller

class turbine_kalman_filter_test(object):

    def __init__(self, file_path):

        # model param:
        self.param = {
            'dt': 0.247,  # sec
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

        self.x_headers = [
            't',
            'PT1',
            'PT2',
            'DP',
            'w',
            'q',
            'g',
            'Tm',
            'I',
            'V',
            'q_pred',
            'w_pred',
            'run_time'
        ]

        self.x = {}
        self.prediction = []

        self.data = Data_processing.turbine_data(file_path)
        self.filter = Turbine_controller.UFK_turbine(self.param, 
                                                     Turbine_SS_model.SS_model(load_model=True, folder_name="SS_models/"))

        # Load the parameters optimized by parameter iddentification module
        self.load_best_param()

    def load_best_param(self):

        save_file_path = "/home/lilly/Energy_Harvester/scripts/Parameter_identification"
        file_name = '/2020_02_21_2param'

        best_param = pickle.load(open(save_file_path + file_name + "_best_param.sav", 'rb'))

        print("\nParameter Loaded: \n")
        for head in best_param.keys():
            self.param[head] = best_param[head]
            print(head + ": " + str("%.3f" % round(self.param[head], 3)))

    def initiate_x(self, data):

        for i, head in enumerate(self.data.transient_col):
            self.x[head] = data[i]

        self.x['q_pred'] = 27
        self.x['w_pred'] = 4000
        self.x['run_time'] = 0

        self.prediction = np.array(
            [list(self.x[head] for head in self.x_headers)])

    def append_x(self):

        self.prediction = np.vstack(
            (self.prediction, list(self.x[head] for head in self.x_headers)))

    def run(self):

        for i, data in enumerate(self.data.transient_list_np):
            self.param['pump_level'] = i
            self.predict(data)
            self.save_predict()

    def predict(self, data):

        self.prediction = []
        self.initiate_x(data[0])
        self.filter.initialize_filter([self.x['q_pred'], self.x['w_pred']],
                                        [4, 10000])
        
        for i in range(1, len(data)):
            for y, head in enumerate(self.data.transient_col):
                if head == 'g':
                    self.x['g'] = max(data[i, y], 0.0)
                elif head == 'I':
                    if abs(self.x['I'] - data[i,y]) >= 0.2:
                        self.x['I'] = data[i,y]
                else:
                    self.x[head] = data[i,y]

            u = np.array([
                self.x['g'],
                self.x['I']
            ])

            y = np.array([
                self.x['PT2'],
                self.x['V']
            ])

            self.start_loop_time = datetime.now()
            pred = self.filter.predict(u,y)
            self.x['run_time'] = (datetime.now() - self.start_loop_time).total_seconds() * 1000

            self.x['q_pred'] = pred['q_pred']
            self.x['w_pred'] = pred['w_pred']

            self.append_x()

    def save_predict(self):

        if len(self.prediction) > 0:

            data_df = pd.DataFrame(self.prediction)
            data_df.columns = self.x_headers
            data_df.to_csv('predict_' + str(self.param['pump_level']) + '.csv')


if __name__ == '__main__':

    import Kalman_filter_test

    data_file_path = "/home/lilly/Energy_Harvester/Processed_data2"

    test = turbine_kalman_filter_test(data_file_path)
    test.run()