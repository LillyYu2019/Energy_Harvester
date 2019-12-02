
import atexit
import serial
import msvcrt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from datetime import datetime

class sensor_recorder(object):

    def __init__(self, print_time = 1, save_rate = 200, plot_time = 1, save_path="out"):

        '''
        data content:
        data_[0] = sensor_read_time //sec
        data_[1] = DP1 //psi
        data_[2] = PT1 //psi
        data_[3] = PT2 //psi
        data_[4] = torque //mNm
        data_[5] = V //V
        data_[6] = I //A
        data_[7] = speed //RPM
        data_[8] = flow_rate //GPM
        data_[9] = GV_angle //deg
        '''

        self.record_ = True

        self.headers_ = ['Time (sec)', 'DP1 (psi)','PT1 (psi)','PT2 (psi)', 'Torque (mNm)','V (V)', 'I (A)', 'Speed (RPM)', 'Flow Rate (GPM)', 'GV Angle (deg)']
        self.identifier_ = ['t', 'DP1', 'PT1', 'PT2', 'tor', 'V', 'I', 'RPM', 'GPM', 'GV']
        self.data_ = []
        for i in range(len(self.headers_)):
            self.data_.append([])

        self.last_print_time_ = datetime.now()
        self.print_time_diff_ = print_time

        self.save_path_ = save_path
        self.save_rate_ = save_rate
        self.current_saves_ = save_rate

        self.show_plot_ = False
        self.last_plot_time_ = datetime.now()
        self.plot_time_diff_ = plot_time
        self.ax_ = []
        self.ax_settings_ = []
        self.plt_obj_ = []
        self.default_vis_setting_ = {'color': "blue", 'linestyle':'-', 'linewidth':1, 'picker': None}

    def read(self, ser):
        '''
        Read serial port data line by line and save in internal variable
        '''
        for i in range(len(self.headers_)):
            read_count = 0

            try:
                line = ser.readline().decode("utf-8")
                line_arr = line.split()
                while line_arr[0] != self.identifier_[i] and read_count < 5:
                    read_count += 1
                    line = ser.readline().decode("utf-8")
                    line_arr = line.split()
                
                if read_count < 5:
                    self.data_[i].append(float(line_arr[-1]))
                else:
                    self.data_[i].append(0.0)
            except:
                self.data_[i].append(0.0)

        return True

    def length(self):
        '''
        Return the number of data point recorded
        '''

        size = len(self.data_[0])

        return size

    def print_to_screen(self):
        '''
        Print latest recorded values to screen at every self.print_time_diff_ seconds
        '''

        if self.length() > 0:
            current_time = datetime.now()
            diff_time = ( current_time - self.last_print_time_).total_seconds()

            if diff_time > self.print_time_diff_:

                for i, dat in enumerate(self.data_):
                    print(self.headers_[i] + ": " + str("%.3f" % round(dat[-1],3)))
                print('\n')

                self.last_print_time_  = current_time

                return diff_time

    def save_data(self, exit = 0):

        '''
        Save data in csv file every self.save_rate_ number of data points.
        Save all current values if 'exit' variable is set to true.
        '''

        if self.length() > self.current_saves_ or exit:

            data_df = pd.DataFrame(self.data_).transpose()
            data_df.columns = self.headers_

            data_df.to_csv(self.save_path_)

            self.current_saves_ = self.length() + self.save_rate_
            
            print("##########################\nData saved!\n##########################\n")

            return True

    def user_input(self, ser, s):

        if s == 'g':
            command = input("Please enter GV angle, pos is cose, neg is open: ")

        elif s == 'i':
            command = input("please enter desired current (0 to 60A): ")

        elif s == 'v':
            command = input("Please enter desired voltage (0 to 30V): ")

        elif s == 's':
            command = input("Please enter sampling time (ms): ")

        elif s == 'p':
            self.record_ = not self.record_
            if not self.record_:
                print("\n##########################\nRecording paused!\n##########################\n")
            else:
                print("\n##########################\nRecording restarted!\n##########################\n")
            return False

        elif s == 's':
            self.save_data(exit=1)
            return False

        elif s =='x':
            if self.length() > 10:
                print("\n##########################\nResetting plot!\n##########################\n")
                for ax_set in self.ax_settings_:
                    ax_set['xmin'] = self.data_[0][-5]
            return False

        elif s == 'l':
            self.show_plot_ = not self.show_plot_
            if self.show_plot_:
                plt.ion()
                plt.show()
            else:
                plt.close('all')
            return False

        else:
            print("Please press:\n  g - GV settings\n  i - current settings\n  v - voltage settings\n  p - pause or restart recording\n s - save current data\n l - show or hide plots\n x - resize plots' x-axis")
            return False

        ser.write((s+'\n'+command).encode("utf-8"))
        print('\n')

        return True

    def resize(self, ax_handle, ax_setting):
        margin = 0.2

        if ax_setting.get("ymin") is not None and ax_setting.get("ymax") is not None:
            y_min = ax_setting.get('ymin')
            y_max = ax_setting.get('ymax')
            y_range = y_max-y_min
            ax_handle.set_ylim([y_min-margin*y_range,y_max+margin*y_range])

        if ax_setting.get("xmin") is not None and ax_setting.get("xmax") is not None:
            x_min = ax_setting.get('xmin')
            x_max = ax_setting.get('xmax')
            x_range = x_max-x_min
            ax_handle.set_xlim([x_min-margin*x_range,x_max+margin*x_range])


    def live_plotter_init(self):

        fig, _ = plt.subplots()
        ax = []
        ax_settings = []

        for i, col in enumerate(self.headers_):
            if i > 0:
                ax.append(plt.subplot(3, 3, i))
                ax_settings.append({'title': col, 'ymin': 0.0, 'ymax': 0.0, 'xmin': 0.0,'xmax': 1.0})

        self.ax_ = ax
        self.ax_settings_ = ax_settings

        return fig

    def live_plotter_update(self):

        if self.length() == 1:
            for i, ax in enumerate(self.ax_):
                ax.set_title(self.ax_settings_[i].get('title'))
                obj, = ax.plot(self.data_[0], self.data_[i+1], self.default_vis_setting_.get('linestyle'), \
                            color=self.default_vis_setting_.get('color'), linewidth=self.default_vis_setting_.get('linewidth'), \
                            picker=self.default_vis_setting_.get('picker'))

                self.plt_obj_.append(obj)

            plt.tight_layout()

        elif self.length() > 1:
            current_time = datetime.now()
            diff_time = ( current_time - self.last_plot_time_).total_seconds()
            
            if diff_time > self.plot_time_diff_:

                xmax = self.data_[0][-1]

                for i, obj in enumerate(self.plt_obj_):
                    obj.set_xdata(self.data_[0])
                    obj.set_ydata(self.data_[i+1])

                    ymax = max(self.data_[i+1])
                    ymin = min(self.data_[i+1])

                    if ymax > self.ax_settings_[i].get('ymax'):
                        self.ax_settings_[i]['ymax'] = ymax 
                    
                    if ymin < self.ax_settings_[i].get('ymin'):
                        self.ax_settings_[i]['ymin'] = ymin 

                    self.ax_settings_[i]['xmax'] = xmax
                    self.resize(self.ax_[i], self.ax_settings_[i])

                self.last_plot_time_  = current_time

                if self.show_plot_:
                    plt.pause(self.plot_time_diff_ *0.5)



if __name__ == '__main__':

    serial_port = 'COM4'
    baud_rate = 9600; #In arduino, Serial.begin(baud_rate)
    write_to_file_path = r"C:\Users\lilly\OneDrive\Documents\1.0_Graduate_Studies\5.0 Energy havester\5.8_code\Energy_Harvester\Data"
    file_name = r"\2019_11_28.csv"

    ser = serial.Serial(serial_port, baud_rate)

    data = sensor_recorder( print_time = 1, save_rate = 200, plot_time = 5, save_path = write_to_file_path + file_name)
    data.live_plotter_init()

    atexit.register(data.save_data, exit=1)

    print("\n##########################\nStart reading!\n##########################\n")

    while True:
        
        if msvcrt.kbhit():
            s =  msvcrt.getch().decode("utf-8")
            data.user_input(ser,s)

        if data.record_:
            data.read(ser)
            data.print_to_screen()
            data.save_data()
            data.live_plotter_update()
                
        


