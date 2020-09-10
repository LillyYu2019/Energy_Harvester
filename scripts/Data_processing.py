import glob
import os

import pandas as pd
import numpy as np

class turbine_data(object):

    def __init__(self, file_path):

        #constants
        self.RPM_to_radpersec = 0.104719755
        self.minor_loss = 0.491 #psi
        self.density_water = 997 #kg/m^3
        self.m3_to_GPM = 15.89806024
        self.psi_to_Pa = 6894.76

        self.file_path = file_path
        self.all_files = glob.glob(os.path.join(file_path, "*.csv")) #make list of paths
        self.file_name = []
        
        self.limits = []

        #steady state data
        self.steady_state_headers = ['w (RPM)', 'I_measured (A)', 'g (deg)','q (GPM)','DP (psi)','Te (mNm)','V (V)']
        #['Speed (RPM)', 'I (A)', 'GV (deg)','Flow Rate (GPM)','DP (psi)','torque (mNm)','V (V)']
        self.steady_state_col = ['w', 'I', 'g', 'q', 'DP', 'Tm','V']
        self.steady_state_df = []
        self.steady_state_np = []
        self.steady_state_len = 0

        #transient data
        self.transient_headers = ['Time (sec)', 'PT1 (psi)', 'PT2 (psi)', 'GV (deg)', 'I (A)', 'Speed (RPM)', 'Flow Rate (GPM)', 'DP (psi)','torque (mNm)', 'V (V)' ]
        self.transient_col = ['t', 'PT1', 'PT2', 'g', 'I', 'w', 'q', 'DP', 'Tm', 'V']
        self.transient_df = []
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

        self.file_name.sort()

        for f in self.file_name:

            dataframe = pd.read_csv(self.file_path + '/' + f + ".csv")

            # dataframe['Speed (rad/s)'] = dataframe['Speed (RPM)']*self.RPM_to_radpersec
            # dataframe['turbine power (w)'] = dataframe['Speed (rad/s)'] * dataframe['torque (mNm)'] / 1000
            # dataframe['fluid power (w)'] = dataframe['Flow Rate (GPM)'] / self.m3_to_GPM * (dataframe['DP (psi)']\
            #                                - self.minor_loss)*self.psi_to_Pa/self.density_water
            # dataframe['turbine eff'] = dataframe['turbine power (w)'] / dataframe['fluid power (w)']

            if "SS" in f:
                if ss_file_loaded:
                    self.steady_state_df = pd.concat([self.steady_state_df, dataframe])
                else:
                    ss_file_loaded = True
                    self.steady_state_df = dataframe
            else:
                self.transient_df.append(dataframe)

            print(f)

    def clean_steady_state_data(self):

        self.steady_state_np = np.array(self.steady_state_df[self.steady_state_headers].sort_values('g (deg)').values)
        self.steady_state_len  = len(self.steady_state_np)
        print("\ntotal num of SS data pts: " +str(self.steady_state_len))

    def clean_transient_data(self):

        total_data = 0

        for df in self.transient_df:
            self.transient_list_np.append(np.array(df[self.transient_headers].values))
            self.transient_list_len.append(len(self.transient_list_np[-1]))
            total_data += self.transient_list_len[-1]

        print("\ntotal num of transient data pts: " +str(total_data))

    def get_limits(self, x_headers):

        self.limits = []
        limits = {}

        for x in x_headers:
            temp = []
            col = self.steady_state_headers[self.steady_state_col.index(x)]
            temp.append(self.steady_state_df[col].min())
            temp.append(self.steady_state_df[col].max())
            self.limits.append(temp)
            limits[x] = temp
        
        return self.limits


if __name__ == '__main__':
    
    from Turbine_SS_model import *
    import pandas as pd

    generate_data = True
    file_path="/home/lilly/Energy_Harvester/Processed_data_sim"


    if generate_data:
        steady_state_model = SS_model(data = None, load_model = True,  folder_name = "SS_model_gap4/")
        data_col = ['q (GPM)', 'g (deg)', 'w (RPM)', 'I_measured (A)', 'Te (mNm)','DP (psi)', 'V (V)']
        data = []

        for q in np.arange(24, 28, 0.5):
            for g in np.arange(0, 9.6 ,1):
                for w in np.arange(2000, 5700 , 200):
                    temp = [q, g, w]
                    inp={
                        'g':g,
                        'q':q,
                        'w':w
                    }
                    out = steady_state_model.predict(inp)
                    I = (out['Tm'] - 10.316) / 46.616
                    V = w / 211.393 - 1.199 * I - 1.333
                    temp.append(I)
                    temp.append(out['Tm'])
                    temp.append(out['DP'])
                    temp.append(V)

                    if V > 0.2 and I < 3.8 and I > 0.8:
                        data.append(temp)

        data_df = pd.DataFrame(data)
        data_df.columns = data_col
        data_df.to_csv(file_path+ '/SS_a.csv')


    data = turbine_data(file_path)

    steady_state_model = SS_model(data = data, load_model = False,  folder_name = "SS_models_sim_gap4/", setting = 'new')
    steady_state_model.train()
    steady_state_model.predict({'DP':17.17, 'w': 4500, 'q':27.059}, prt_to_screen = True) #pt2 = 34.57
    steady_state_model.predict({'Tm':175.84, 'DP': 17.17, 'q':27.059}, prt_to_screen = True)
    steady_state_model.predict({'w':4500, 'g': 7.5, 'q':27.059}, prt_to_screen = True)
    # print(data.get_limits(['w', 'I', 'g', 'q', 'DP', 'Tm','V']))
    steady_state_model.save_models()
    steady_state_model.plot_surface()
    # steady_state_model.plot_efficiency()