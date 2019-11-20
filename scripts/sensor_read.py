
import atexit
import serial
import msvcrt

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from datetime import datetime

class sensor_recorder(object):

    def __init__(self, print_rate = 1, save_rate = 200, save_path="out"):

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

        self.headers_ = ['Time (sec)', 'DP1 (psi)','PT1 (psi)','PT2 (psi)', 'Torque (mNm)','V (V)', 'I (A)', 'Speed (RPM)', 'Flow Rate (GPM)', 'GV Angle (deg)']
        self.data_ = []
        for i in range(10):
            self.data_.append([])

        self.last_print_time_ = datetime.now()
        self.print_time_diff_ = 1/print_rate
        
        self.save_path_ = save_path
        self.save_rate_ = save_rate
        self.current_saves_ = save_rate

    def read(self, ser):

        for i in range(10):
            line = float(ser.readline().decode("utf-8"))
            self.data_[i].append(line)

        return i

    def length(self):

        size = len(self.data_[0])

        return size

    def print_to_screen(self):

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

        if self.length() > self.current_saves_ or exit:

            data_df = pd.DataFrame(self.data_).transpose()
            data_df.columns = self.headers_

            data_df.to_csv(self.save_path_)

            self.current_saves_ += self.save_rate_
            
            print("##########################\nData saved!\n##########################\n")

            return True

    def write_to_serial(self, ser, s):

        if s == 'g':
            command = input("Please enter GV angle, pos is cose, neg is open: ")
        elif s == 'i':
            command = input("please enter desired current (0 to 60A): ")
        elif s == 'v':
            command = input("Please enter desired voltage (0 to 30V): ")
        else:
            print("Please press g for GV, i for current, or v for voltage settings.\n")
            return False

        ser.write((s+'\n'+command).encode("utf-8"))
        print('\n')

        return True

if __name__ == '__main__':

    serial_port = 'COM3'
    baud_rate = 9600; #In arduino, Serial.begin(baud_rate)
    write_to_file_path = r"C:\Users\lilly\OneDrive\Documents\1.0_Graduate_Studies\5.0 Energy havester\5.8_code\Energy_Harvester\Data"
    file_name = r"\2019_11_20.csv"

    ser = serial.Serial(serial_port, baud_rate)

    data = sensor_recorder( print_rate = 0.5, save_rate = 100, save_path = write_to_file_path + file_name)

    atexit.register(data.save_data, exit=1)

    print("\n##########################\nStart reading!\n##########################\n")

    while True:
        
        if msvcrt.kbhit():
            s =  msvcrt.getch().decode("utf-8")
            data.write_to_serial(ser,s)

        data.read(ser)
        data.print_to_screen()
        data.save_data()
                
        


