
from serial import Serial
from scipy.signal import butter, lfilter, lfilter_zi
import pandas as pd
import matplotlib.pyplot as plt

class lowpass_filter(object):

    def __init__(self, cutoff, fs, order = 5):

        self.normal_cutoff = cutoff / (0.5 * fs)
        self.b, self.a = butter(order, self.normal_cutoff, btype='low', analog=False)
        print(self.b)
        print(self.a)
        self.z = lfilter_zi(self.b, self.a)
        self.count = 0

    def filter(self, data):

        res, self.z = lfilter(self.b, self.a,[data], zi=self.z)
        self.count += 1

        if self.count > 40:
            return res[0]
        else:
            return data

class sensor_recorder(object):

    def __init__(self, input_port, param):

        self.ser = Serial(input_port, 9600)
        self.param = param

        self.identifier = ['PT1', 'PT2', 'V', 'I', 'w', 'q', 'g', 'dt']
        self.x = {}
        self.invalid_data = True

        for head in self.identifier:
            self.x[head] = 0
        
        self.pt1_filter = lowpass_filter(0.5, 4.54)
        self.pt2_filter = lowpass_filter(0.5, 4.54)
        self.I_filter = lowpass_filter(0.3, 4.54)
        self.V_filter = lowpass_filter(0.3, 4.54)
        self.q_filter = lowpass_filter(0.1, 4.54)
        self.w_filter = lowpass_filter(1, 4.54)

    def read(self):
        '''
        Read serial port data line by line and save in internal variable
        '''
        self.invalid_data = False
        for i in self.identifier:
            read_count = 0

            try:
                line = self.ser.readline().decode()

                line_arr = line.split()

                if len(line_arr) < 1:
                    self.invalid_data = True
                    continue

                while line_arr[0] != i and read_count < 5:
                    read_count += 1
                    line = self.ser.readline().decode("utf-8")
                    line_arr = line.split()

                if read_count < 5:
                    val = float(line_arr[-1])
                    self.x[i] = val
                else:
                    self.invalid_data = True

            except:
                self.invalid_data = True
                continue

        if self.x['g'] < 0 :
            self.x['g'] = 0


        self.x['g_measured'] = self.x['g']
        self.x['V_measured'] = self.V_filter.filter(self.x['V']) + 0.2
        self.x['PT1_measured'] = self.pt1_filter.filter(self.x['PT1'])
        self.x['PT2_measured'] = self.pt2_filter.filter(self.x['PT2'])
        self.x['I_measured'] = self.I_filter.filter(self.x['I']) + 0.1
        self.x['q_measured'] = self.q_filter.filter(self.x['q'])
        self.x['w_measured'] = self.w_filter.filter(self.x['w'])
        
        self.x['DP'] = self.x['PT1'] - self.x['PT2']
        self.x['Te'] = self.param['torque_constant'] * self.x['I'] + self.param['torque_resistance']
        self.x['Tm'] = self.x['Te']
        
        return self.x




if __name__ == '__main__':


    from Turbine_sensors import *
    
    fs = 1
    cutoff = 0.5
    f = lowpass_filter(cutoff, fs)
    
    dataframe = pd.read_csv("/home/lilly/Energy_Harvester/Data" + '/' + '2020_08_09_ff_0809' + ".csv")

    filtered = []
    original = dataframe['w (RPM)'].values
    time = dataframe['t (sec)'].values

    for p in original:
        filtered.append(f.filter(p))

    plt.plot(time, original, 'b-', label = 'original')
    plt.plot(time, filtered, 'g-', label = 'filtered')
    plt.legend()
    plt.show()

