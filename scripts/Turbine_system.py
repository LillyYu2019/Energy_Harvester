import atexit
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from pynput import keyboard


import Turbine_sensors
import Turbine_SS_model
import Turbine_controller
import Turbine_actuators
import Turbine_dynamic_model


class turbine_system(object):

    def __init__(self, input_port, output_port, write_to_file_path, file_name, mode="sim", x_init={}):

        self.mode = mode

        # model param:
        self.param = {
            'dt': 0.2,  # sec
            'Tw': 1.88,
            'H': 0.266,
            'torque_constant': 46.541,  # mNm/A
            'torque_resistance': 10.506,  # mNm
            'speed_constant': 216.943 * 0.104719755,  # rad/s/V
            'internal_resistance': 2.434,
            'minor_loss': 0.491,  # psi
            'density_water': 997,  # kg/m^3
            'm3_to_GPM': 15.89806024,
            'psi_to_Pa': 6894.76,
            'RPM_to_radpersec': 0.104719755
        }
        # model variables
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
            'V',
            'P_f',
            'P_t',
            'P_e',
            'eff_t',
            'eff_g'
        ]
        self.x_units = [
            'sec',
            'psi',
            'psi',
            'psi',
            'RPM',
            'GPM',
            'deg',
            'mNm',
            'V',
            'deg',
            'A',
            'W',
            'W',
            'W',
            '',
            ''
        ]
        self.h_headers = {
            'g_des',
            'I_des'
        }
        self.x_ref_headers = {
            'DP_ref',
            'w_ref'
        }

        self.x = {}
        self.h = {}
        self.x_ref = {}

        self.data = []
        self.SS_data = []
        self.length = 0
        self.length_SS = 0

        self.SS_start_length = 0
        self.pause = False
        self.taking_SS_data = False
        self.save_path = write_to_file_path + file_name

        #timer variables:
        self.print_time = 1 #sec
        self.loop_time = 0.5 #sec
        self.last_print_time = datetime.now()
        self.start_loop_time = datetime.now()
        self.run_time = 0 #sec

        # create all subsystems:
        if self.mode == "live":
            self.sensors = Turbine_sensors.sensor_recorder(input_port)
        if self.mode == "sim":
            self.dynamic_model = Turbine_dynamic_model.dynamic_model(
                self.param)
            self.actuator = Turbine_actuators.actuators(self.param)
        self.SS_model = Turbine_SS_model.SS_model(load_model=True)
        self.controller = Turbine_controller.controller_main(self.mode, output_port, self.SS_model, self.param)

        self.init(x_init)

    def init(self, x_init):

        if self.mode == 'live':
            while self.sensors.invalid_data:
                x_init = self.sensors.read()

        self.update_x(x_init)
        self.update_metrics()
        self.append_x()
        print("\n##########################\nStart reading!\n##########################\n")

    def run(self):

        if self.pause or (datetime.now() - self.start_loop_time).total_seconds() < self.loop_time:
            return

        self.start_loop_time = datetime.now()

        if self.mode == 'live':

            # read data from arduino, update if reading is valid
            x = self.sensors.read()
            if not self.sensors.invalid_data:
                self.update_x(x)
                self.update_metrics()
                self.append_x()
                self.average_steady_state_data()

        if self.mode == "sim":

            # update g_des and I_des based on controller output
            self.Turbine_controller_step()

            # update actual g and I based on actuator dynamics
            self.Turbine_actuators_step()

            # update h and Tm given q, g and w using Turbine SS model output
            self.Turbine_SS_model_step()

            # update Te, w, q, and all other performance metrics using turbine dynamic model
            self.Turbine_dynamic_model_step()

            # calculate all performance metrics and append the new x
            self.update_metrics()
            self.append_x()

        self.print_to_screen()
        self.run_time = (datetime.now() - self.start_loop_time).total_seconds()

    def update_x(self, x):

        for head in x.keys():
            self.x[head] = x[head]

    def append_x(self):

        self.data.append(list(self.x[head] for head in self.x_headers))
        self.length += 1

    def Turbine_controller_step(self):

        self.h = self.controller.step(self.x_ref, self.x)

    def Turbine_actuators_step(self):

        self.update_x(self.actuator.step(self.h, self.x))

    def Turbine_SS_model_step(self):

        inp = {
            'Flow Rate (GPM)': self.x['q'],
            'GV (deg)': self.x['g'],
            'Speed (RPM)': self.x['w']
        }
        out = self.SS_model.predict(inp)
        self.update_x({
            'DP': out['DP (psi)'], 
            'Tm': out['torque (mNm)'],
            'PT1': out['DP (psi)']+self.x['PT2']
            })

    def Turbine_dynamic_model_step(self):

        self.update_x(self.dynamic_model.step(self.x, self.data))

    def update_metrics(self):

        self.x['P_e'] = self.x['V'] * self.x['I']
        self.x['P_t'] = self.x['Tm'] * self.x['w'] * \
            self.param['RPM_to_radpersec'] / 1000.0
        self.x['P_f'] = self.x['q'] / self.param['m3_to_GPM'] * \
            (self.x['DP'] - self.param['minor_loss']) * \
            self.param['psi_to_Pa']/self.param['density_water']
        try:
            self.x['eff_t'] = self.x['P_t'] / self.x['P_f']
        except:
            self.x['eff_t'] = 0.0
        try:
            self.x['eff_g'] = self.x['P_e']/self.x['P_t']
        except:
            self.x['eff_g'] = 0.0

    def average_steady_state_data(self, SS_sample_time=5):

        if self.taking_SS_data:
            if self.data[-1][0] - self.data[self.SS_start_length-1][0] > SS_sample_time:
                self.SS_data.append(
                    list(np.mean(np.array(self.data[self.SS_start_length-1:]), axis=0)))
                self.taking_SS_data = False
                self.length_SS += 1
                print(
                    "##########################\nSS Data recorded!\n##########################\n")

    def print_to_screen(self):

        if self.length > 2:

            if (datetime.now() - self.last_print_time).total_seconds() > self.print_time:

                print('\n')
                for head in self.x_ref.keys():
                    print(head + ": " +
                          str("%.3f" % round(self.x_ref[head], 3)))

                for head in self.h.keys():
                    print(head + ": " +
                          str("%.3f" % round(self.h[head], 3)))
                
                print('\n')
                for i, head in enumerate(self.x_headers):
                    print(head + " (" + self.x_units[i]+"): " +
                          str("%.3f" % round(self.x[head], 3)))
                print("Sample time: " +
                      str("%.1f" % round((self.data[-1][0] - self.data[-2][0])*1000, 1)))


                self.last_print_time = datetime.now()

    def save_data(self):

        headers = []
        for i, head in enumerate(self.x_headers):
            headers.append(head + " (" + self.x_units[i]+")")

        if self.length > 0:

            data_df = pd.DataFrame(self.data)
            data_df.columns = headers
            data_df.to_csv(self.save_path + '.csv')

            print("##########################\nData saved!\n##########################\n")

        if self.length_SS > 0:

            data_df = pd.DataFrame(self.SS_data)
            data_df.columns = headers
            data_df.to_csv(self.save_path + '_SS.csv')

            print(
                "##########################\nSS Data saved!\n##########################\n")

    def user_input(self, key):

        try:
            s = key.char
        except:
            s = ""

        if s == 'r':
            self.pause = True
            DP_ref = float(input("Enter desired DP (psi): ")[1:])
            w_ref = float(input("Enter desired speed (rpm): "))

            self.x_ref = {
                'DP_ref': DP_ref,
                'w_ref': w_ref
            }
            self.pause = False
            return True

        elif s == 'p':
            self.pause = not self.pause
            if self.pause:
                print(
                    "\n##########################\nRecording paused!\n##########################\n")
            else:
                print(
                    "\n##########################\nRecording restarted!\n##########################\n")
            return True

        elif s == 'd':
            if not self.taking_SS_data:
                print(
                    "\n##########################\nStarted SS data recording!\n##########################\n")
                self.taking_SS_data = True
                self.SS_start_length = self.length
            else:
                print(
                    "\n##########################\nCurrent SS data recording not done yet!\n##########################\n")
            return True

        elif s == 's':
            self.save_data()
            return True

        else:
            # print("Please press:\n\
            #     p - pause or restart recording\n\
            #     s - save current data\n\
            #     d - take steady state data")
            return True


if __name__ == '__main__':

    # custom libraries
    import Turbine_system

    # Turbine system initialization
    input_port = '/dev/ttyACM0'
    output_port = '/dev/ttyACM1'
    write_to_file_path = "/home/lilly/Energy_Harvester/Data"
    file_name = "/test"

    x_init = {'t': 0.0,
              'PT1': 38.7,
              'PT2': 31.4,
              'DP': 7.33,
              'w': 3942,
              'q': 27.32,
              'g': 0,
              'Tm': 56.6,
              'Te': 56.6,
              'I': 0.99,
              'V': 16.3,
              'g_des': 0,
              'I_des': 1}

    pico_turbine = Turbine_system.turbine_system(input_port,
                                                 output_port,
                                                 write_to_file_path,
                                                 file_name,
                                                 mode="sim",
                                                 x_init=x_init)

    # save data on exit
    atexit.register(pico_turbine.save_data)

    # set up keyboard interaction (non-blocking)
    listener = keyboard.Listener(on_press=pico_turbine.user_input)
    listener.start()

    while True:
        pico_turbine.run()
