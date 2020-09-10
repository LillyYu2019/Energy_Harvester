import atexit
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

from datetime import datetime
from pynput import keyboard

import Turbine_sensors
import Turbine_SS_model
import Turbine_controller
import Turbine_actuators
import Turbine_dynamic_model


if __name__ == '__main__':

    # custom libraries
    import Turbine_system

    # Turbine system initialization
    input_port = '/dev/ttyACM1'
    output_port = '/dev/ttyACM0'
    write_to_file_path = "/home/lilly/Energy_Harvester/Data"
    file_name = "/zthesis_low_pid_gap4"

    x_init = {'PT1': 38.723,
              'PT2': 31.3898,
              'DP': 7.333,
              'w': 3942.096,
              'q': 27.324,
              'g': 0.0,
              'Tm': 56.5669,
              'Te': 56.5669,
              'I': 0.992,
              'V': 16.317}

    pico_turbine = Turbine_system.turbine_system(input_port,
                                                 output_port,
                                                 write_to_file_path,
                                                 file_name,
                                                 mode="sim",
                                                 x_init=x_init)

    # save data on exit
    atexit.register(pico_turbine.exit_script)

    # set up keyboard interaction (non-blocking)
    listener = keyboard.Listener(on_press=pico_turbine.user_input)
    listener.start()

    # ani = animation.FuncAnimation(pico_turbine.fig, pico_turbine.run, interval=500)
    # plt.show()
    while True:
        pico_turbine.run(1)

class turbine_system(object):

    def __init__(self, input_port, output_port, write_to_file_path, file_name, mode="sim", x_init={}):

        self.mode = mode

        # model param:
        self.param = {
            'loop_time': {'fuzzy': 5, 'model': 1, 'ff': 1, 'observer': 0.5},
            'global_controller' : '', #model or fuzzy
            'low_level_controller': ['pid'], # ff and/or pid
            'controller_on': True,
            'dt': 0.1,  # sec
            't_total': 550 +450, #sec
            'pump_level': 0,
            'q_pred' : 25.567,
            'w_pred' : 3258,
            'P_init' : [4, 10000],
            'I_sat' : [1.0, 10],
            'g_sat' : [0.0, 8.4],
            'w_sat' : [2500, 5500], #used by global controller
            'PT2_constants':  [-1.9, 0, 0.0452], #[1.7181, 0, 0.0509], 
            'PT1_constants':  [[129.8, -3.37],
                               [128, -3.365],
                               [127.3, -3.353],
                               [130.5, -3.375],
                               [128.5, -3.355],],

            '''
            [[114.319, -2.83],
                               [114.43, -2.7633],
                               [113.95, -2.9321],
                               [113.27, -2.9841],
                               [110.16, -2.4658],],
            '''
            'Tw': 1.325,
            'H': 0.353,
            'torque_constant': 46.616,  # mNm/A
            'torque_resistance': 10.316,  # mNm
            'speed_constant': 211.393,  # rad/s/V
            'internal_resistance': 1.199,
            'V_intercept': 1.325,
            'minor_loss': 0.491,  # psi
            'm3_to_GPM': 15850.323114,
            'psi_to_Pa': 6894.76,
            'RPM_to_radpersec': 0.104719755,
        }

        #timer variables:
        self.time_temp = 0
        self.print_time = 1 #sec
        self.loop_time = 0.02 #sec
        self.start_time = datetime.now()
        self.last_print_time = self.start_time
        self.start_loop_time = self.start_time
        self.run_time = 0 #sec

        self.h = {}
        self.x_ref = {}
        self.x = {}
        self.x_prev = {}
        self.data = []
        self.SS_data = []
        self.length = 0
        self.length_SS = 0

        self.SS_start_length = 0
        self.pause = False
        self.taking_SS_data = False
        self.user_entered_ref = False
        self.save_path = write_to_file_path + file_name

        #reference signal generation
        self.PT2_schedule = [
            [30, 31, 3400],
            [100, 30, 4000],
            [200, 29.2, 4200],
            [300, 30, 3600],
            [400, 31.2, 3600],
        ]

        '''
        original:
            [30, 31.2, 4000],
            [100, 29, 4800],
            [200, 27, 5300],
            [300, 28, 4800],
            [400, 29, 4800],
        gap2:
            [30, 29, 6000],
            [100, 28, 6500],
            [200, 27, 7000],
            [300, 28, 6500],
            [400, 29, 6000],

        gap4
                    [30, 31, 3400],
            [100, 30, 4000],
            [200, 29.2, 4200],
            [300, 30, 3600],
            [400, 31.2, 3600],
        '''

        self.q_schedule = [
            [0, 0],
            [100, 1],
            [120, 2],
            [140, 3],
            [160, 2],
            [180, 4],
            [200, 1],
            [220, 3],
            [240, 2],
            [260, 1],
            [280, 0],
            [300, 3],
            [320, 0],
            ]

        for i, temp in enumerate(self.q_schedule):
            self.q_schedule[i][0] = temp[0]+450

        self.PT2_constants_schedule = [
            [340, [-2, 0, 0.046]],
            [360, [-2.3, 0, 0.047]],
            [380, [-1.5, 0, 0.044]],
            [400, [-1.7, 0, 0.043]],
            [420, [-1.9, 0, 0.0452]],
            [440, [-1.9, 0, 0.0435]],
            [460, [-1.8, 0, 0.046]],
            [480, [-1.9, 0, 0.0452]],
            [500, [-2.1, 0, 0.0475]],
            [520, [-2, 0, 0.046]],
            [540, [-1.9, 0, 0.0452]]
        ]

        for i, temp in enumerate(self.PT2_constants_schedule):
            self.PT2_constants_schedule[i][0] = temp[0]+450

        #noise level:
        self.noise_level = {
            'V': 0.05,
            'PT1': 0.03,
            'PT2': 0.03,
            'g': 0.001,
            'I': 0.001,
            'q': 0.1,
            'w': 0.1
        }

        #plotting variables
        self.new_message = False
        self.titles = [
            'Pressure (psi)',
            'Angular Speed (rpm)',
            'GV angle (deg)',
            'Pressure Drop (psi)',
            'Flow rate (GPM)',
            'Torque (mNm)',
            'Current (I)',
            'Voltage (V)',
            'Power (W)',
            'Efficiency'
        ]
        self.obj_titles = [
            ['PT2_ref','PT2', 'PT1', 'PT2_measured', 'PT1_measured'],
            ['w_ref', 'w', 'w_pred'],
            ['g_des', 'g'],
            ['DP_ref', 'DP'],
            ['q', 'q_pred'],
            ['Tm', 'Te'],
            ['I_des', 'I', 'I_measured'],
            ['V'],
            ['P_f', 'P_t', 'P_e'],
            ['eff_t', 'eff_g']
        ]
        self.vis_setting = [
            ["blue", '.'],
            ["green", '.'],
            ["red", '.'],
            ["orange", '-'],
            ['yellow', '-']
        ]

        # Load the parameters optimized by parameter iddentification module
        self.load_best_param()
        
        # model variables
        self.x_headers = [
        #simulated true state variables: x_sim
        't', 'PT1', 'PT2', 'DP', 'w', 'q', 'Tm', 'Te', 'V', 'dt',
        # desired reference variables: x_ref
        'DP_ref', 'w_ref', 'PT2_ref',
        #estimated state variables: x_pred
        'q_pred', 'w_pred',
        #controller desired output: h_des
        'g_des', 'I_des',
        #actual actuator output: h
        'g', 'I',
        #metrics calculated using measured state variables: x_metrics
        'P_f', 'P_t', 'P_e', 'eff_t', 'eff_g', 'eff', 'P_e_max', 'best_w',
        #filtered variables in live or noise added variables in sim
        'V_measured', 'PT1_measured', 'PT2_measured', 'g_measured', 'I_measured', 'q_measured', 'w_measured'
        ]

        self.x_units = [
        #simulated true state variables: x_sim
        'sec', 'psi', 'psi', 'psi', 'RPM', 'GPM', 'mNm', 'mNm', 'V', 'sec',
        #desired reference variables: x_ref
        'psi', 'RPM', 'psi',
        #estimated state variables: x_pred
        'GPM', 'RPM',
        #controller desired output: h_des
        'deg', 'A',
        #actual actuator output: h
        'deg', 'A',
        #metrics calculated using measured state variables: x_metrics
        'W', 'W', 'W', '', '', '','W','RPM',
        #filtered variables in live or noise added variables in sim
        'V', 'psi', 'psi', 'deg', 'A', 'GPM', 'RPM'
        ]

        # Create all subsystems:
        if self.mode == "live":
            self.sensors = Turbine_sensors.sensor_recorder(input_port, self.param)
        
        if self.mode == "sim":
            self.dynamic_model = Turbine_dynamic_model.dynamic_model(self.param)
            self.actuator = Turbine_actuators.actuators(self.param)
            self.SS_model_sim = Turbine_SS_model.SS_model(load_model=True, folder_name = "SS_models_sim_gap4/", setting='new')

        # Load initial conditions
        self.init(x_init)

        self.SS_model = Turbine_SS_model.SS_model(load_model=True, folder_name = "SS_models_sim/",setting='new')
        self.controller = Turbine_controller.controller_main(self.mode, output_port, self.SS_model, self.param, self.x.copy())



    def run(self, z):
        
        if self.x['t'] > self.param['t_total']:
            self.pause = True

        if self.pause or  (datetime.now() - self.start_loop_time).total_seconds() < self.loop_time:
            return

        self.start_loop_time = datetime.now()

        # generate reference signal
        self.generate_reference()

        if self.mode == 'live':

            # read data from arduino, update if reading is valid
            self.sensor_read()

        if self.mode == "sim":

            # update actual g and I based on actuator dynamics
            self.Turbine_actuators_step()

            # update h and Tm given q, g and w using Turbine SS model output
            self.Turbine_SS_model_step()

            # update Te, w, q, using turbine dynamic model
            self.Turbine_dynamic_model_step()

            #add sensor noise to simulated states variables
            self.add_sensor_noise()
        
        # update g_des and I_des based on controller output
        self.Turbine_controller_step()

        # calculate all performance metrics and append the new x
        self.update_turbine_metrics()
        self.append_x()

        # print and plot current data
        self.run_time = (datetime.now() - self.start_loop_time).total_seconds()
        self.print_to_screen()

    def init(self, x_init):

        if self.mode == 'live':
            while self.sensors.invalid_data:
                print('initial reading invalid!')
                x_init = self.sensors.read()

        for head, val in x_init.items():
            self.x[head] = val

        self.x['t'] = 0
        self.x['dt'] = 0
        self.x['DP_ref'] = self.x['DP']
        self.x['w_ref'] = self.x['w']
        self.x['PT2_ref'] = self.x['PT2']

        self.x['g_des'] = self.x['g']
        self.x['I_des'] = self.x['I']

        self.x['q_pred'] = self.param['q_pred']
        self.x['w_pred'] = self.param['w_pred']

        self.x['loop_time'] = 0
        self.x['P_e_max'] = 0
        self.x['best_w'] = 0

        if self.mode == 'sim':
            self.add_sensor_noise()

        self.update_turbine_metrics()

        self.data = np.array([list(self.x[head] for head in self.x_headers)])
        self.init_plots()

        print("\n##########################\nStart reading!\n##########################\n")

    def update_x(self, x):

        for head, val in x.items():
            self.x[head] = val

    def append_x(self):

        self.data = np.vstack((self.data, list(self.x[head] for head in self.x_headers)))
        self.new_message = True
        self.length += 1

    def load_best_param(self):

        save_file_path = "/home/lilly/Energy_Harvester/scripts/Parameter_identification"
        file_name = '/2020_02_21_2param'

        best_param = pickle.load(open(save_file_path + file_name + "_best_param.sav", 'rb'))

        print("\nParameter Loaded: \n")
        for head, val in best_param.items():
            self.param[head] = val
            print(head + ": " + str("%.3f" % round(val, 3)))

    def sensor_read(self):

        x = self.sensors.read()
        while self.sensors.invalid_data:
            print("invalid reading")
            x = self.sensors.read()
        self.update_x(x)
        self.average_steady_state_data()

    def generate_reference(self):

        self.x_prev = self.x.copy()

        if self.mode == 'sim':
            self.x['t'] = self.x['t'] + self.param['dt']
            self.x['dt'] = self.param['dt']
        elif self.mode == 'live':
            current_time = (datetime.now() - self.start_time).total_seconds()
            self.param['dt'] = current_time - self.x['t']
            self.x['t'] = current_time

        if self.user_entered_ref:
            self.update_x(self.x_ref)
            return

        if len(self.PT2_schedule) > 0:
            if self.x['t'] > self.PT2_schedule[0][0]:
                self.x_ref['PT2_ref'] = self.PT2_schedule[0][1]
                if len(self.param['global_controller']) < 1:
                    self.x_ref['w_ref'] = self.PT2_schedule[0][2]
                self.PT2_schedule.pop(0)
    
        if len(self.q_schedule) > 0:
            if self.x['t'] > self.q_schedule[0][0]:
                self.param['pump_level'] = self.q_schedule[0][1]
                self.q_schedule.pop(0)
        
        if len(self.PT2_constants_schedule) > 0:
            if self.x['t'] > self.PT2_constants_schedule[0][0]:
                self.param['PT2_constants'] = self.PT2_constants_schedule[0][1]
                self.PT2_constants_schedule.pop(0)

    def Turbine_actuators_step(self):

        self.update_x(self.actuator.step(self.h, self.x_prev.copy()))

    def Turbine_SS_model_step(self):

        inp = {
            'q': self.x_prev['q'],
            'g': self.x_prev['g'],
            'w': self.x_prev['w']
        }

        out = self.SS_model_sim.predict(inp)
        
        self.update_x(out)

    def Turbine_dynamic_model_step(self):

        self.update_x(self.dynamic_model.step(self.x_prev.copy()))

    def add_sensor_noise(self):
        
        for key, val in self.noise_level.items():
            self.x[key +'_measured'] = self.x[key] + np.random.normal(0,val,1)[0]

    def Turbine_controller_step(self):

        self.h, x_ref, x_pred = self.controller.step(self.x_ref.copy(), self.x.copy(), self.h.copy())

        self.update_x(self.h)
        self.update_x(x_ref)
        self.update_x(x_pred)

    def update_turbine_metrics(self):

        self.x['P_e'] = self.x['V'] * self.x['I']
        self.x['P_t'] = self.x['Tm'] * self.x['w'] * \
            self.param['RPM_to_radpersec'] / 1000.0
        self.x['P_f'] = self.x['q'] / self.param['m3_to_GPM'] * \
            (self.x['DP'] - self.param['minor_loss']) * self.param['psi_to_Pa']
        if self.x['P_f'] > 0:
            self.x['eff_t'] = self.x['P_t'] / self.x['P_f']
        else:
            self.x['eff_t'] = 0
        if self.x['P_t'] > 0:
            self.x['eff_g'] = self.x['P_e']/self.x['P_t']
        else:
            self.x['eff_g'] = 0
        self.x['eff'] = self.x['eff_t']*self.x['eff_g']

        inp = {
            'DP': self.x['PT1'] - self.x['PT2'],
            'q': self.x['q']
        }
        

        if self.x['t'] - self.time_temp > 0.2:
            self.time_temp = self.x['t']
            best_P = 0
            best_w = 4000
            for I in np.arange(self.param['I_sat'][0], self.param['I_sat'][1], 0.1):

                inp['Tm'] = self.param['torque_constant']* I + self.param['torque_resistance']
                out = self.SS_model_sim.predict(inp)

                V = out['w'] / self.param['speed_constant'] - self.param['internal_resistance'] * I - self.param['V_intercept']
                P_e = V * I
                # print(inp)
                # print(out['w'])
                # print(P_e)
                if P_e > best_P:
                    best_P = P_e
                    best_w = out['w']

            self.x['P_e_max'] = best_P
            self.x['best_w'] = best_w

    def evaluate_performance(self):

        if self.length > 0:
            num = int(min(self.param['t_total']/self.param['dt'], len(self.data)))
            data_df = pd.DataFrame(self.data[0:num])
            data_df.columns = self.x_headers

            IAE_p = self.param['dt'] * (data_df['PT2'] - data_df['PT2_ref']).abs().sum()
            IAE_w = self.param['dt'] * (data_df['w'] - data_df['w_ref']).abs().sum()
            total_energy_recovered = self.param['dt'] * data_df['P_e'].sum()
            possible_energy_recovered = self.param['dt'] * data_df['P_e_max'].sum()

            print("IAE_p: " +
                        str("%.1f" % round(IAE_p, 1)))
            print("IAE_w: " +
                        str("%.1f" % round(IAE_w, 1)))
            print("total_energy_recovered (J): " +
                        str("%.1f" % round(total_energy_recovered, 1)))
            print("possible_energy_recovered (J): " +
                        str("%.1f" % round(possible_energy_recovered, 1)))

    def average_steady_state_data(self, SS_sample_time=5):

        if self.taking_SS_data:
            if self.data[-1][0] - self.data[self.SS_start_length-1][0] > SS_sample_time:
                self.SS_data.append(
                    list(np.mean(np.array(self.data[self.SS_start_length-1:]), axis=0)))
                self.taking_SS_data = False
                self.length_SS += 1
                print("#####################\nSS Data recorded!\n#######################\n")

    def print_to_screen(self):

        if self.length > 2:

            if (datetime.now() - self.last_print_time).total_seconds() > self.print_time:
                
                print('\n')
                for i, head in enumerate(self.x_headers):
                    try:
                        print(head + " (" + self.x_units[i]+"): " +
                            str("%.3f" % round(self.x[head], 3)))
                    except:
                        print(self.x[head])


                if self.mode == "live":
                    print("Sample time (s): " +
                        str("%.1f" % round((self.data[-1][0] - self.data[-2][0])*1000, 1)))
                else:
                    print("run time (ms): " +
                        str("%.1f" % round(self.run_time*1000, 1)))

                self.last_print_time = datetime.now()

    def init_plots(self):

        self.fig = plt.figure(tight_layout=True, figsize=(20, 14))
        gs = gridspec.GridSpec(4, 3)

        self.axes = []
        self.axes.append(self.fig.add_subplot(gs[0:2,0]))
        self.axes.append(self.fig.add_subplot(gs[2:4,0]))
        for i in range(4):
            self.axes.append(self.fig.add_subplot(gs[i,1]))
        for i in range(4):
            self.axes.append(self.fig.add_subplot(gs[i,2]))

        self.objs = []
        for i, ax in enumerate(self.axes):
            ax.set_title(self.titles[i])
            self.objs.append([])
            for j, var in enumerate(self.obj_titles[i]):
                self.objs[i].append(ax.plot([], [], self.vis_setting[j][1],
                                                    color=self.vis_setting[j][0],
                                                    label=var)[0])
            ax.legend(loc="upper right")

    def plot_once(self):

        for i, plot in enumerate(self.objs):
            for j, obj in enumerate(plot):
                obj.set_data(self.data[:,0], self.data[:, self.x_headers.index(self.obj_titles[i][j])])
            self.axes[i].relim()
            self.axes[i].autoscale_view()
    
    def update_plots(self):

        if self.length > 2 and self.new_message:
            self.plot_once()
            self.new_message = False

    def save_data(self):

        headers = []
        for i, head in enumerate(self.x_headers):
            headers.append(head + " (" + self.x_units[i]+")")

        if self.length > 0:

            data_df = pd.DataFrame(self.data)
            data_df.columns = headers
            data_df.to_csv(self.save_path + '.csv')

            self.fig.savefig(self.save_path+'_plot.png', dpi=150)

            print("##########################\nData saved!\n##########################\n")

        if self.length_SS > 0:

            data_df = pd.DataFrame(self.SS_data)
            data_df.columns = headers
            data_df.to_csv(self.save_path + '_SS.csv')

            print(
                "##########################\nSS Data saved!\n##########################\n")

    def exit_script(self):

        if self.mode == 'live':
            self.shown_down_turbine()

        self.evaluate_performance()
        self.plot_once()

        self.save_data()

        plt.show()

    def shown_down_turbine(self):

        commend = "C\t"+ str(1.0) +"\t"+ str(0.00)
        print("shutting down turbine")
        print(commend)
        self.controller.ser.write(commend.encode()) 

    def user_input(self, key):

        try:
            s = key.char
        except:
            s = ""

        if s == 'r':
            self.pause = True

            try:
                PT2_ref = float(input("Enter desired PT2 (psi): ")[1:])
            except:
                PT2_ref = float(input("Enter desired PT2 (psi): "))
            w_ref = float(input("Enter desired speed (rpm): "))

            self.x_ref['PT2_ref'] = PT2_ref
            self.x_ref['w_ref'] = w_ref

            print(self.x_ref)
            self.user_entered_ref = True
            self.pause = False

            return True

        elif s == 'i':

            self.pause = True
            self.param['controller_on'] = False

            try:
                self.h['g_des'] = float(input("Enter desired g (deg): ")[1:])
            except:
                self.h['g_des'] = float(input("Enter desired g (deg): "))
            self.h['I_des'] = float(input("Enter desired I (A): "))
            self.update_x(self.h)

            commend = "C\t"+ str(round(self.h['I_des'],3)) +"\t"+ str(round(self.h['g_des'],3))
            print(self.h)
            self.controller.ser.write(commend.encode()) 

            self.pause = False

            return True

        elif s == 'o':
            self.param['controller_on'] = not self.param['controller_on']
            if self.param['controller_on'] :
                if 'pid' in self.param['low_level_controller']:
                    self.controller.low_level_controler[0].clear(self.x['I_measured'], self.x['g_measured'])
                print(
                    "\n##########################\Controller started!\n##########################\n")
            else:
                print(
                    "\n##########################\Controller paused!\n##########################\n")
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

        elif s == 'k':
            self.param['controller_on'] = False
            self.shown_down_turbine()
            return True

        else:
            # print("Please press:\n\
            #     p - pause or restart recording\n\
            #     s - save current data\n\
            #     d - take steady state data")
            return True


