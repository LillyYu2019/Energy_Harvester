
import serial
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from datetime import datetime

class sensor_recorder(object):

    def __init__(self, print_rate = 1):

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

        self.headers_ = ['time (sec)', 'DP1 (psi)','PT1 (psi)','PT2 (psi)', 'Torque (mNm)','V (V)', 'I (A)', 'Speed (RPM)', 'Flow Rate (GPM)', 'GV Angle (deg)']
        self.data_ = []
        for i in range(10):
            self.data_.append([])

        self.last_print_time_ = datetime.now()
        self.print_diff_time_ = 1/print_rate

    def write(self, line, i):

        self.data_[i].append(line)
        return i

    def length(self):

        size = len(self.data_[0])

        for i in self.data_:
            if len(i) < size:
                size = len(i)
        
        return size

    def print_to_screen(self):

        if self.length() > 0:
            current_time = datetime.now()
            diff_time = ( current_time - self.last_print_time_).total_seconds()

            if diff_time > self.print_diff_time_:

                for i, dat in enumerate(self.data_):
                    print(self.headers_[i] + ": " + str("%.3f" % round(dat[-1],3)))
                print('\n')

                self.last_print_time_  = current_time

                return diff_time
        



if __name__ == '__main__':

    serial_port = 'COM3'
    baud_rate = 9600; #In arduino, Serial.begin(baud_rate)
    write_to_file_path = "20191118.txt"

    ser = serial.Serial(serial_port, baud_rate)

    data = sensor_recorder( print_rate= 0.5)

    print("start reading \n")
    while True:
        
        for i in range(10):

            line = float(ser.readline().decode("utf-8"))
            data.write(line, i)

        data.print_to_screen()
            


