
import glob
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import pygmo as pg
from smt.surrogate_models import KRG
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)

#base class that read, clean, and store all experimental data
class turbine_data(object):

    def __init__(self, file_path):

        #constants
        self.RPM_to_radpersec_ = 0.104719755
        self.minor_loss_ = 0.491 #psi
        self.density_water_ = 997 #kg/m^3
        self.m3_to_GPM_ = 15.89806024
        self.psi_to_Pa_ = 6894.76

        self.all_files = glob.glob(os.path.join(file_path, "*.csv")) #make list of paths
        self.file_name_ = []
        self.steady_state_df_ = []
        self.transient_df_ = []

        #steady state data
        self.steady_state_headers_ = ['Speed (rad/s)','GV','Flow Rate (GPM)','DP (psi)','torque (mNm)','turbine eff']
        self.steady_state_np_ = []

        #transient data
        self.transient_headers_ = ['Time (sec)', 'GV', 'I (A)', 'Speed (rad/s)', 'Flow Rate (GPM)', 'DP (psi)','torque (mNm)', 'V (V)' ]
        self.transient_np_ = []

        self.read_all()
        self.clean_steady_state_data()
        self.clean_transient_data()
        
    def read_all(self):

        print("\nloading files: ")
        ss_file_loaded = False

        for f in self.all_files:

            file_name = os.path.splitext(os.path.basename(f))[0]  # Getting the file name without extension
            self.file_name_.append(file_name)

            dataframe = pd.read_csv(f)

            dataframe['Speed (rad/s)'] = dataframe['Speed (RPM)']*self.RPM_to_radpersec_
            dataframe['turbine power (w)'] = dataframe['Speed (rad/s)'] * dataframe['torque (mNm)'] / 1000
            dataframe['fluid power (w)'] = dataframe['Flow Rate (GPM)'] / self.m3_to_GPM_ * (dataframe['DP (psi)']-self.minor_loss_)*self.psi_to_Pa_/self.density_water_
            dataframe['turbine eff'] = dataframe['turbine power (w)'] / dataframe['fluid power (w)']

            if "SS" in file_name:
                if ss_file_loaded:
                    self.steady_state_df_ = pd.concat([self.steady_state_df_, dataframe])
                else:
                    ss_file_loaded = True
                    self.steady_state_df_ = dataframe
            else:
                self.transient_df_.append(dataframe)

            print(file_name)

    def clean_steady_state_data(self):

        self.steady_state_np_ = np.array(self.steady_state_df_[self.steady_state_headers_].sort_values('GV').values)
        print("\ntotal num of SS data pts: " +str(len(self.steady_state_np_)))

    def clean_transient_data(self):

        total_data = 0

        for df in self.transient_df_:
            self.transient_np_.append(np.array(df[self.transient_headers_].values))
            total_data += len(self.transient_np_[-1])

        print("\ntotal num of transient data pts: " +str(total_data))
        


if __name__ == '__main__':
    
    file_path=r"C:\Users\lilly\OneDrive\Documents\1.0_Graduate_Studies\5.0 Energy havester\5.8_code\Energy_Harvester\Processed_data"

    experimental_data = turbine_data(file_path)