
import atexit
import serial
import msvcrt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from datetime import datetime


class sensor_recorder(object):

    def __init__(self, print_time=1, save_rate=200, plot_time=1, save_path="out"):
        '''
        data content:
        data_[0] = sensor_read_time //sec
        data_[1] = PT1 //psi
        data_[2] = PT2 //psi
        data_[3] = torque //mNm
        data_[4] = V //V
        data_[5] = I //A
        data_[6] = speed //RPM
        data_[7] = flow_rate //GPM
        data_[8] = GV_angle //deg
        '''

        self.record_ = True
        self.taking_steady_state_data_ = False
        self.SS_start_length_ = 0

        self.headers_ = ['Time (sec)', 'PT1 (psi)', 'PT2 (psi)','torque (mNm)', 'V (V)', 'I (A)', 'Speed (RPM)', 'Flow Rate (GPM)', 'GV (deg)', 'DP (psi)']
        
        self.identifier_ = ['t', 'PT1', 'PT2', 'tor', 'V', 'I', 'RPM','GPM', 'GV']

        self.data_steady_state_ = []
        self.data_ = []
        self.length = 0
        self.I_offset_ = 0.0
        self.V_offset_ = 0.0
        
        for i in range(len(self.headers_)):
            self.data_.append([])
            self.data_steady_state_.append([])

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
        self.default_vis_setting_ = {
            'color': "blue", 'linestyle': '-', 'linewidth': 1, 'picker': None}

    def read(self, ser):
        '''
        Read serial port data line by line and save in internal variable
        '''
        for i in range(len(self.identifier_)):
            read_count = 0

            try:
                line = ser.readline().decode("utf-8")
                line_arr = line.split()
                while line_arr[0] != self.identifier_[i] and read_count < 5:
                    read_count += 1
                    line = ser.readline().decode("utf-8")
                    line_arr = line.split()

                if read_count < 5:
                    if self.identifier_[i] == "I":
                        val = float(line_arr[-1]) + self.I_offset_
                    elif self.identifier_[i] == "V":
                        val = float(line_arr[-1]) + self.V_offset_
                    else:
                        val = float(line_arr[-1])
                    self.data_[i].append(val)
                else:
                    print("Warning - read count > 5, read this: " + line)
                    self.data_[i].append(0.0)

            except:
                print("Warning read exception: " + self.identifier_[i])
                self.data_[i].append(0.0)

            if self.identifier_[i] == "PT2":
                self.data_[len(self.identifier_)].append(self.data_[1][-1]-self.data_[2][-1])

            self.length += 1

        return True

    def length(self):
        '''
        Return the number of data point recorded
        '''
        return self.length

    def print_to_screen(self):
        '''
        Print latest recorded values to screen at every self.print_time_diff_ seconds
        '''

        if self.length() > 1:
            current_time = datetime.now()
            diff_time = (current_time - self.last_print_time_).total_seconds()

            if diff_time > self.print_time_diff_:

                for i, dat in enumerate(self.data_):
                    print(self.headers_[i] + ": " +
                          str("%.3f" % round(dat[-1], 3)))
                print("Sample time: " +
                          str("%.1f" % round((self.data_[0][-1] - self.data_[0][-2])*1000, 1)))
                print('\n')

                self.last_print_time_ = current_time

                return diff_time

    def average_steady_state_data(self, SS_sample_time = 5):
        '''
        take an average reading over a sample time period and write to the
        steady state data array to be saved seperately
        '''
        if self.taking_steady_state_data_:
            if self.data_[0][-1] - self.data_[0][self.SS_start_length_-1] > SS_sample_time:
                self.data_steady_state_[0].append(self.data_[0][self.SS_start_length_-1])
                for i in range(len(self.headers_)-1):
                    self.data_steady_state_[i+1].append(np.average(self.data_[i+1][self.SS_start_length_-1:]))
                self.taking_steady_state_data_ = False
                print("##########################\nSS Data saved!\n##########################\n")

    def save_data(self, exit=0):
        '''
        Save data in csv file every self.save_rate_ number of data points.
        Save all current values if 'exit' variable is set to true.
        '''

        if self.length() > self.current_saves_ or exit:

            data_df = pd.DataFrame(self.data_).transpose()
            data_df.columns = self.headers_

            data_df.to_csv(self.save_path_ + '.csv')

            self.current_saves_ = self.length() + self.save_rate_

            if self.data_steady_state_[0]:

                data_df_SS = pd.DataFrame(self.data_steady_state_).transpose()
                data_df_SS.columns = self.headers_

                data_df_SS.to_csv(self.save_path_+'_SS.csv')

            print("##########################\nData saved!\n##########################\n")

            return True

    def user_input(self, ser, s):

        if s == 'g':
            command = input(
                "Please enter GV angle, pos is cose, neg is open: ")

        elif s == 'i':
            print("current I offset: " + str("%.3f" % self.I_offset_))
            offset = float(input("please enter desired current offset (0 to 60A): "))
            if offset < 60.0:
                self.I_offset_ = offset
            return False

        elif s == 'v':
            print("current V offset: " + str("%.3f" % self.V_offset_))
            offset = float(input("Please enter desired voltage offset (0 to 30V): "))
            if offset < 30.0:
                self.V_offset_ = offset
            return False

        elif s == 't':
            list_of_sensors = ""
            for i, sensor in enumerate(self.identifier_[1:]):
                list_of_sensors = list_of_sensors + \
                    str(i) + " - " + sensor + "\n"

            command1 = input(list_of_sensors + "Please enter sensor number: ")
            command2 = input("Please enter sampling time (ms): ")
            command = command1 + "\n" + command2

        elif s == 'm':
            command = input("Please enter desired motor speed (0 to 10): ")

        elif s == 'p':
            self.record_ = not self.record_
            if not self.record_:
                print(
                    "\n##########################\nRecording paused!\n##########################\n")
            else:
                print(
                    "\n##########################\nRecording restarted!\n##########################\n")
            return False

        elif s == 's':
            if not self.taking_steady_state_data_:
                print(
                    "\n##########################\nStarted SS data recording!\n##########################\n")
                self.taking_steady_state_data_ = True
                self.SS_start_length_ = self.length()
            else:
                print(
                    "\n##########################\nCurrent SS data recording not done yet!\n##########################\n")

            return False

        elif s == 'x':
            if self.length() > 10:
                print(
                    "\n##########################\nResetting plot!\n##########################\n")
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
            print("Please press:\n\
                g - GV settings\n\
                i - current settings\n\
                v - voltage settings\n\
                t - sampling time settings\n\
                m - change motor speed\n\
                p - pause or restart recording\n\
                s - save current data\n\
                l - show or hide plots\n\
                x - resize plots' x-axis")
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
            ax_handle.set_ylim([y_min-margin*y_range, y_max+margin*y_range])

        if ax_setting.get("xmin") is not None and ax_setting.get("xmax") is not None:
            x_min = ax_setting.get('xmin')
            x_max = ax_setting.get('xmax')
            x_range = x_max-x_min
            ax_handle.set_xlim([x_min-margin*x_range, x_max+margin*x_range])

    def live_plotter_init(self):

        fig, _ = plt.subplots()
        ax = []
        ax_settings = []

        for i, col in enumerate(self.identifier_):
            if i > 0:
                ax.append(plt.subplot(3, 3, i))
                ax_settings.append(
                    {'title': col, 'ymin': 0.0, 'ymax': 0.0, 'xmin': 0.0, 'xmax': 1.0})

        self.ax_ = ax
        self.ax_settings_ = ax_settings

        return fig

    def live_plotter_update(self):

        if self.length() == 1:
            for i, ax in enumerate(self.ax_):
                ax.set_title(self.ax_settings_[i].get('title'))
                obj, = ax.plot(self.data_[0], self.data_[i+1], self.default_vis_setting_.get('linestyle'),
                               color=self.default_vis_setting_.get('color'), linewidth=self.default_vis_setting_.get('linewidth'),
                               picker=self.default_vis_setting_.get('picker'))

                self.plt_obj_.append(obj)

            plt.tight_layout()

        elif self.length() > 1:
            current_time = datetime.now()
            diff_time = (current_time - self.last_plot_time_).total_seconds()

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

                self.last_plot_time_ = current_time

                if self.show_plot_:
                    plt.pause(self.plot_time_diff_ * 0.5)


if __name__ == '__main__':

    serial_port = 'COM4'
    baud_rate = 9600  # In arduino, Serial.begin(baud_rate)
    write_to_file_path = r"C:\Users\lilly\OneDrive\Documents\1.0_Graduate_Studies\5.0 Energy havester\5.8_code\Energy_Harvester\Data"
    file_name = r"\2020_02_22"

    ser = serial.Serial(serial_port, baud_rate)

    data = sensor_recorder(print_time=1, save_rate=200,
                           plot_time=5, save_path=write_to_file_path + file_name)
    #data.live_plotter_init()

    atexit.register(data.save_data, exit=1)

    print("\n##########################\nStart reading!\n##########################\n")

    while True:

        if msvcrt.kbhit():
            s = msvcrt.getch().decode("utf-8")
            data.user_input(ser, s)

        if data.record_:
            data.read(ser)
            data.print_to_screen()
            data.average_steady_state_data()
            data.save_data()
           # data.live_plotter_update()
