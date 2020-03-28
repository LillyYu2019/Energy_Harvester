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

    def get_limits(self, x_headers):

        self.limits = []

        for x in x_headers:
            temp = []
            temp.append(self.steady_state_df[x].min())
            temp.append(self.steady_state_df[x].max())
            self.limits.append(temp)
        
        return self.limits


if __name__ == '__main__':
    
    from Turbine_SS_model import *

    file_path=r"C:\Users\lilly\OneDrive\Documents\1.0_Graduate_Studies\5.0 Energy havester\5.8_code\Energy_Harvester\Processed_data2"

    data = turbine_data(file_path)

    steady_state_model = SS_model(data = data, load_model = False)
    steady_state_model.train()
    steady_state_model.predict({'DP (psi)':19.9, 'Speed (rad/s)':567, 'Flow Rate (GPM)':25.0}, prt_to_screen = True)
    steady_state_model.predict({'GV (deg)': 6.5, 'Flow Rate (GPM)': 25.82, 'torque (mNm)': 132.44342}, prt_to_screen = True)
    steady_state_model.save_models()
    steady_state_model.plot_surface()