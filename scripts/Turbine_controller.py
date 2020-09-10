from serial import Serial
from datetime import datetime
import numpy as np
import math 
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
import skfuzzy as fuzz
from skfuzzy import control as fuzz_ctrl

class observer_base(object):

    def __init__(self, param):

        '''
        state space model:
        x[k+1] = f(x[k],u[k]) + w
        y[k+1] = h(x[k]) + v

        E[wwT] = Q
        E[vvT] = R

        x = [ q  w ]
        u = [ g  I h1 h2]
        y = [ DP  V]


        '''
        self.param = param

class UKF_turbine(observer_base):
    
    def __init__(self, param, SS_model, dim_x = 2, dim_u = 2, dim_y = 2):

        super().__init__(param)

        self.SS_model = SS_model
        self.last_time = 0
        self.last_commend = {}

        points = MerweScaledSigmaPoints(n=2, alpha=.1, beta=2, kappa=1)
        self.ukf = UKF(dim_x = dim_x, dim_z = dim_y,
                       fx = self.fx, hx=self.hx, dt=param['dt'], points=points)

        self.ukf.R = np.diag([0.0672, #Pressure transducer (DP) noise 0.0672
                              0.0187])    #Voltage noise 0.00187
        self.ukf.Q = np.diag([0.15**2,     #Pressure disturbance
                              0.2**2])    #Torque disturbance

        self.initialize_filter([param['q_pred'], param['w_pred']], param['P_init'])
    
    def initialize_filter(self, x_init, P_init):

        self.ukf.x = np.array(x_init)
        self.ukf.P = np.diag(P_init)

    def fx(self, x, dt, u):

        q = x[0]
        w = x[1]

        g = u[0]
        I = u[1]
        h1 = u[2]
        h2 = u[3]

        Tw = self.param['Tw']
        Tg = self.param['H'] 

        inp = {
            'q': q,
            'g': g,
            'w': w
        }

        out = self.SS_model.predict(inp)

        ht = out['DP']
        Tm = out['Tm']

        Te = self.param['torque_constant']*I + self.param['torque_resistance']

        q_next = q + 1/Tw * (h1 - ht - h2) * dt
        w_next = w + 1/Tg * (Tm - Te) * dt / self.param['RPM_to_radpersec']

        return np.array([q_next, w_next])

    def hx(self, x, u):

        q = x[0]
        w = x[1]
        g = u[0]
        I = u[1]
    
        inp = {
            'q': q,
            'g': g,
            'w': w
        }
        
        out = self.SS_model.predict(inp)
        

        ht = out['DP']

        V = w / self.param['speed_constant'] - self.param['internal_resistance'] * I - self.param['V_intercept']

        return np.array([ht, V])

    def predict(self, u, y, t):

        if t - self.last_time < self.param['loop_time']['observer']:
            return self.last_commend

        self.ukf.predict(u=u)
        self.ukf.update(y, u=u)

        x_pred = {
            'q_pred' : self.ukf.x[0],
            'w_pred' : self.ukf.x[1]
        }

        self.last_time = t
        self.last_commend = x_pred

        return x_pred

class global_controller_base(object):

    def __init__(self, param, x_current):

        self.param = param
        self.last_w = x_current['w_ref']
        self.start_time = 0

    def sat(self, val, min_val, max_val):

        if val < min_val:
            val = min_val
        if val > max_val:
            val = max_val

        return val

    def disable_global_controller(self,  x_ref, x_current):

        if abs(x_current['PT2_measured'] - x_ref['PT2_ref']) > 0.2:
            return True
        else:
            return False

class optimal_tracking(global_controller_base):

    def __init__(self, SS_model, param, x_current):

        super().__init__(param, x_current)

        self.SS_model = SS_model
        self.best_P = 0
        self.last_inp = {
            'DP': 0,
            'q': 0
        }

    def step(self, x_ref, x_current, x_pred):
        
        if (x_current['t'] - self.start_time) < self.param['loop_time']['model'] or self.disable_global_controller(x_ref, x_current):
            x_ref['w_ref'] = self.last_w
            return x_ref

        self.last_inp = {
            'DP': x_current['PT1_measured'] - x_current['PT2_measured'],
            'q': x_current['q_measured']
        }
        best_P = 0
        best_w = 4000

        for I in np.arange(self.param['I_sat'][0], self.param['I_sat'][1], 0.1):

            self.last_inp['Tm'] = self.param['torque_constant']* I + self.param['torque_resistance']
            out = self.SS_model.predict(self.last_inp)

            V = out['w'] / self.param['speed_constant'] - self.param['internal_resistance'] * I - self.param['V_intercept']
            P_e = V * I

            if P_e > best_P:
                best_P = P_e
                best_w = out['w']

        #     print(out['w'])
        #     print(P_e)
        # print()

        if abs(best_P - self.best_P) < 0.5 or abs(self.last_w - best_w) < 20:
            x_ref['w_ref'] = self.last_w
        else:
            best_w = self.sat(best_w, self.param['w_sat'][0], self.param['w_sat'][1])
            self.best_P = best_P
            self.last_w = best_w
            print("Newwwwwwwwwwww best")
            print(best_P)
            print(best_w)
            x_ref['w_ref'] = best_w
        
        self.start_time = x_current['t']

        return x_ref

class fuzz_optimal_tracking(global_controller_base):

    def __init__(self, param, x_current):

        super().__init__(param,x_current)

        self.k1 = 1
        self.k2 = 1
        self.k3 = 1
        self.w_prev = 0
        self.P_prev = 0
        self.DP_prev = 0
        self.PT2_ref = x_current['PT2_ref']
        self.swtich = False
        self.start_time_p = 0
        
        self.delta_w_fuzz_set = np.array([
          -200, -50, -20, 0, 20, 50, 200
        ])

        self.delta_P_fuzz_set = np.array([
          -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1
        ])

        self.init_fuzz_table()

    def init_fuzz_table(self):

        delta_w = fuzz_ctrl.Antecedent(self.delta_w_fuzz_set, 'delta_w')
        delta_p = fuzz_ctrl.Antecedent(self.delta_P_fuzz_set, 'delta_p')
        delta_w_ref = fuzz_ctrl.Consequent(self.delta_w_fuzz_set, 'delta_w_ref')

        names = ['BN', 'MN', 'SN', 'Z', 'SP', 'MP', 'BP']

        for i, val in enumerate(names):
            z=max(i-1, 0)
            j=min(i+1,len(names)-1)
            delta_w[val] = fuzz.trimf(delta_w.universe,[self.delta_w_fuzz_set[z], self.delta_w_fuzz_set[i],self.delta_w_fuzz_set[j]])
            delta_p[val] = fuzz.trimf(delta_p.universe,[self.delta_P_fuzz_set[z], self.delta_P_fuzz_set[i],self.delta_P_fuzz_set[j]])
            delta_w_ref[val] = fuzz.trimf(delta_w_ref.universe,[self.delta_w_fuzz_set[z], self.delta_w_fuzz_set[i],self.delta_w_fuzz_set[j]])
        
        fuzz_table = np.array([
            ['BP', 'BP', 'MP', 'Z', 'MN', 'BN', 'BN'],
            ['BP', 'MP', 'SP', 'Z', 'SN', 'MN', 'BN'],
            ['MP', 'SP', 'SP', 'Z', 'SN', 'SN', 'MN'],
            ['BN', 'MN', 'SN', 'Z', 'SP', 'MP', 'BP'],
            ['MN', 'SN', 'SN', 'Z', 'SP', 'SP', 'MP'],
            ['BN', 'MN', 'SN', 'Z', 'SP', 'MP', 'BP'],
            ['BN', 'BN', 'MN', 'Z', 'MP', 'BP', 'BP']])

        rules = []
        for i, w in enumerate(names):
            for j, P in enumerate(names):
                rules.append(fuzz_ctrl.Rule(delta_w[w] & delta_p[P], delta_w_ref[fuzz_table[i,j]]))

        w_ref_opt_ctrl = fuzz_ctrl.ControlSystem(rules)
        self.w_ref_opt = fuzz_ctrl.ControlSystemSimulation(w_ref_opt_ctrl, flush_after_run=20)
    
    def step(self, x_ref, x_current, x_pred):
        
        if self.disable_global_controller(x_ref, x_current):
            self.w_prev = x_current['w']
            self.P_prev = x_current['eff_t'] * x_current['eff_g']
            self.DP_prev= x_current['DP']
            x_ref['w_ref'] = self.last_w
            self.start_time = x_current['t']
            return x_ref

        if (x_current['t'] - self.start_time_p) > 27 and self.swtich:
            print("switching up")
            x_ref['w_ref'] = x_current['w_ref'] + 100
            self.swtich = False
            self.start_time_p = x_current['t']
            print(x_ref['w_ref'])
        elif (x_current['t'] - self.start_time_p) > 27 and not self.swtich:
            print("switching down")
            x_ref['w_ref'] = x_current['w_ref'] - 100
            self.swtich = True
            self.start_time_p = x_current['t']
            print(x_ref['w_ref'])
        elif (x_current['t'] - self.start_time) > self.param['loop_time']['fuzzy']:
            
            if x_current['I_des'] <=1.01:
                x_ref['w_ref'] = x_current['w_ref'] - 200
            elif x_current['I_des'] >= 3.39:
                x_ref['w_ref'] = x_current['w_ref'] + 200
            elif x_current['g_des'] >= 8.39:
                x_ref['w_ref'] = x_current['w_ref'] + 200
            else:
                dP = abs(self.DP_prev - x_current['DP'])/self.param['loop_time']['fuzzy']
                print("dp")
                print(dP)
                if self.w_prev <= 0 or dP > 0.1 or self.disable_global_controller(x_ref, x_current):
                    self.w_prev = x_current['w']
                    self.P_prev = x_current['eff_t'] * x_current['eff_g']
                    self.DP_prev= x_current['DP']
                    x_ref['w_ref'] = self.last_w
                    self.start_time = x_current['t']
                    return x_ref

                inp = {
                    'delta_w' : (x_current['w'] - self.w_prev) * self.k1,
                    'delta_p' : (x_current['eff_t'] * x_current['eff_g'] - self.P_prev) * self.k2
                }
                
                self.w_ref_opt.inputs(inp)
                self.w_ref_opt.compute()
                    
                w_ref = (self.w_ref_opt.output['delta_w_ref'] * self.k3) + self.w_prev
                x_ref['w_ref'] = self.sat(w_ref, self.param['w_sat'][0], self.param['w_sat'][1])

                print(inp)
                print(self.w_ref_opt.output['delta_w_ref'])

            print(x_ref['w_ref'])
            self.last_w = x_ref['w_ref']
            self.w_prev = x_current['w']
            self.P_prev = x_current['eff_t'] * x_current['eff_g']
            self.DP_prev= x_current['DP']
            self.start_time = x_current['t']

        else:
            x_ref['w_ref'] = self.last_w

        self.last_w = x_ref['w_ref']
        return x_ref


class Low_level_controller_base(object):

    def __init__(self, param):

        self.param = param

    def sat(self, val, min_val, max_val):

        if val < min_val:
            val = min_val
        if val > max_val:
            val = max_val

        return val

class feed_forward(Low_level_controller_base):

    def __init__(self, SS_model, param):

        super().__init__(param)

        self.SS_model = SS_model
        self.commend = {}
        self.last_time = 0
        self.last_commend = {}

    def step(self, x_ref, x_current, x_pred):
        
        if x_current['t'] - self.last_time < self.param['loop_time']['ff']:
            return self.last_commend

        if True:
            self.c1 = self.param['PT1_constants'][self.param['pump_level']]
            self.c2 = self.param['PT2_constants']

            q = math.sqrt((x_ref['PT2_ref'] - self.c2[0]) / self.c2[2])
            PT1 = self.c1[1] * q + self.c1[0]
            
            if q < 20 or q > 35 or PT1 - x_ref['PT2_ref'] < 0:
                return self.last_commend

            inp = {
                'DP': PT1 - x_ref['PT2_ref'] ,
                'w': x_ref['w_ref'],
                'q': q
            }
        else:

            inp = {
                'DP': x_current['PT1_measured'] - x_ref['PT2_ref'] ,
                'w': x_ref['w_ref'],
                'q': x_current['q_measured']
            }

        out = self.SS_model.predict(inp)
        
        self.commend = {
            'g_des': out['g'], 
            'I_des': (out['Tm'] - self.param['torque_resistance'] ) / self.param['torque_constant']
        }

        self.last_time = x_current['t']
        self.last_commend = self.commend

        print("ff controller")
        print(inp)
        print(out)

        return self.commend

class feedback(Low_level_controller_base):

    def __init__(self, param):

        super().__init__(param)

        '''

        MIMO PID setup:
        [g = [ Kp11 + Ki11/s  Kp12 + Ki12/s  * [ PT2- PT2_ref
         I]    Kp21 + Ki21/s  Kp11 + Ki22/s]     w - w_ref ] 

        '''
        self.I_offset = 0.0
        self.g_offset = 0.0

        self.Kp = np.array([
            [0.2, 0],
            [0, 0.0003]
        ])
        self.Ki = np.array([
            [0.3, 0],
            [0, 0.0003]
        ])
        self.e = np.zeros((2,1))
        self.sum_e = np.zeros((2,1))

        if 'ff' in self.param['low_level_controller']:
            g_e_sat = (8) / self.Ki[0,0]
            I_e_sat = (3.6) / self.Ki[1,1]
        else:
            g_e_sat = (self.param['g_sat'][1] - 0) / self.Ki[0,0]
            I_e_sat = (self.param['I_sat'][1] - 0) / self.Ki[1,1]

        self.sum_e_sat = np.array([
            [g_e_sat],
            [I_e_sat]
        ])

    def clear(self, I_current, g_current):

        self.I_offset = I_current
        self.g_offset = g_current

        self.e = np.zeros((2,1))
        self.sum_e = np.zeros((2,1))

    def step(self, x_ref, x_current, x_pred):

        self.e[0,0] = x_current['PT2_measured'] - x_ref['PT2_ref'] 
        self.e[1,0] = x_current['w_measured'] - x_ref['w_ref'] 

        if (x_current['I_des'] <= self.param['I_sat'][0]+0.001 and self.e[1,0] < 0) or (x_current['I_des'] >= self.param['I_sat'][1]-0.001 and self.e[1,0] > 0):
            print("I sat capped")
        else:
            self.sum_e[1,0] += self.e[1,0]*self.param['dt']
            

        if (x_current['g_des'] < self.param['g_sat'][0]+0.001 and self.e[0,0] < 0 ) or (x_current['g_des'] >= self.param['g_sat'][1]-0.001 and self.e[0,0] > 0):
            print("g sat capped")
        else:
            self.sum_e[0,0] += self.e[0,0]*self.param['dt']
        
        for i in range(len(self.e)):
            self.sum_e[i,0] = self.sat(self.sum_e[i,0], -self.sum_e_sat[i,0], self.sum_e_sat[i,0])

        h = np.matmul(self.Kp, self.e) + np.matmul(self.Ki, self.sum_e)
        
        commend = {
            'g_des' : h[0,0] + self.g_offset,
            'I_des' : h[1,0] + self.I_offset,
        }

        return commend

class controller_main(object):

    def __init__(self, mode, output_port, SS_model, param, x_current):

        self.last_cmd_t = 0
        self.mode = mode
        self.SS_model = SS_model
        self.x_previous = {}
    
        if mode == "live":
            self.ser = Serial(output_port, 9600)

        #constants
        self.param = param
        self.out_min = {'g_des': param['g_sat'][0], 'I_des': param['I_sat'][0]}
        self.out_max = {'g_des': param['g_sat'][1], 'I_des': param['I_sat'][1]}

        #init all controllers
        self.UKF = UKF_turbine(self.param, SS_model)

        if param['global_controller'] == 'fuzzy':
            self.global_controller = fuzz_optimal_tracking(param, x_current)
        elif param['global_controller'] == 'model':
            self.global_controller = optimal_tracking(SS_model, param, x_current)
        
        self.low_level_controler = []
        if 'pid' in param['low_level_controller']:
            self.low_level_controler.append(feedback(param))
        if 'ff' in param['low_level_controller']:
            self.low_level_controler.append(feed_forward(SS_model, param))

    
    def sub(self, dict1, dict2):

        new_dict ={}

        for v in dict1:
            if v in dict2:
                new_dict[v] = dict1[v] - dict2[v]
            else:
                new_dict[v] = dict1[v]

        return new_dict

    def add(self, dict1, dict2):

        new_dict ={}

        for v in dict1:
            if v in dict2:
                new_dict[v] = dict1[v] + dict2[v]
            else:
                new_dict[v] = dict1[v]

        return new_dict

    def saturate(self, out, lower, upper):

        for y in out:
            if out[y] < lower[y]:
                out[y] = lower[y]
            if out[y] > upper[y]:
                out[y] = upper[y]
        
        return out

    def observer_step(self, x_current):
        
        if 't' not in self.x_previous:
            self.x_previous = x_current.copy()
            return {'w_pred':self.param['w_pred'], 'q_pred':self.param['q_pred']}

        u = [
            self.x_previous['g_measured'],
            self.x_previous['I_measured'],
            self.x_previous['PT1_measured'],
            self.x_previous['PT2_measured']
        ]
        y = [
            x_current['PT1_measured'] - x_current['PT2_measured'],
            x_current['V_measured']
        ]
        
        self.x_previous = x_current.copy()

        x_pred = self.UKF.predict(u,y, x_current['t'])

        return x_pred

    def global_controller_step(self, x_ref, x_current, x_pred):

        x_ref = self.global_controller.step(x_ref, x_current, x_pred)

        return x_ref

    def low_level_controller_step(self, x_ref, x_current, x_pred, h_prev):

        h_des = {
            'g_des' : 0,
            'I_des' : 0
        }

        # low level controller
        for controller in self.low_level_controler:
            h_des = self.add(h_des, controller.step(x_ref, x_current, x_pred))

        #saturate output
        h_des = self.saturate(h_des, self.out_min, self.out_max)
        
        return h_des

    def step(self, x_ref, x_current, h_prev):
        
        #Predict state variables using observer
        x_pred = self.observer_step(x_current)

        #If reference is not set, no controller action
        if 'PT2_ref' not in x_ref or not self.param['controller_on']:
            return {}, x_ref, x_pred

        #calculate desired perssure drop from desired downstream pressure
        x_ref['DP_ref'] = x_current['PT1_measured'] - x_ref['PT2_ref']

        #Call global optimal controller to compute desired speed that will track maximum efficiency
        if 'w_ref' not in x_ref and len(self.param['global_controller']) > 0:
            x_ref = self.global_controller_step(x_ref, x_current, x_pred)

        #Compute command giving current and reference x
        h_des = self.low_level_controller_step(x_ref, x_current, x_pred, h_prev)

        if x_current['t'] - self.last_cmd_t > 0.5:
            self.last_cmd_t  = x_current['t']
            #Send command to Arduino if in live mode
            if self.mode == "live":
                if "I_des" in h_des:
                    commend = "C\t"+ str(round(h_des['I_des'],3)) +"\t"+ str(round(h_des['g_des'],3))
                    print(commend)
                    self.ser.write(commend.encode()) 

            return h_des, x_ref, x_pred

        else:
            return {}, x_ref, x_pred

        


    if __name__ == "__main__":
        
        param = {
            'dt': 0.1,  # sec
            't_total': 105, #sec
            'pump_level': 0,
            'q_pred' : 26,
            'w_pred' : 3000,
            'P_init' : [4, 10000],
            'PT2_constants': [-1.8998, 0, 0.0452],
            'PT1_constants' : [[129.8, -3.373],
                               [125.5, -3.3638],
                               [118.59, -3.2444]],
            'Tw': 1.88,
            'H': 0.266,
            'torque_constant': 46.541,  # mNm/A
            'torque_resistance': 10.506,  # mNm
            'speed_constant': 213,  # rad/s/V
            'internal_resistance': 1.67,
            'V_intercept': 1.3,
            'minor_loss': 0.491,  # psi
            'density_water': 997,  # kg/m^3
            'm3_to_GPM': 15.89806024,
            'psi_to_Pa': 6894.76,
            'RPM_to_radpersec': 0.104719755
        }
        x_pred = {}
        x_current = {}

        x_pred['w_pred'] = 4400
        x_current['P_e'] = 20
        x_current['DP'] = 1
        x_current['PT2_measured'] = 1

        ctrl = fuzz_optimal_tracking(param)
        x = ctrl.step( {'PT2_ref':1}, x_current, x_pred)
        print(x)
        x_pred['w_pred'] = 4400 - 11.69
        x_current['P_e'] = 20 - 0.193/1.5
        x_current['DP'] = 1
        x = ctrl.step( {'PT2_ref':1}, x_current, x_pred)
        print(x)
