from serial import Serial

class feed_forward(object):

    def __init__(self, SS_model, param):

        self.SS_model = SS_model
        self.param = param

    def step(self, x_ref, x_current):

        inp = {
            'DP (psi)': x_ref['DP_ref'],
            'Speed (RPM)': x_ref['w_ref'],
            'Flow Rate (GPM)': x_current['q']
        }
        out = self.SS_model.predict(inp)
        commend = {
            'g': out['GV (deg)'], 
            'I': (out['torque (mNm)'] - self.param['torque_resistance']) / self.param['torque_constant']
        }
        
        return commend

class controller_main(object):

    def __init__(self, mode, output_port, SS_model, param):

        self.mode = mode
        self.SS_model = SS_model
        if mode == "live":
            self.ser = Serial(output_port, 9600)

        #constants
        self.param = param
        self.out_min = {'g': 0.0, 'I': 0.8}
        self.out_max = {'g': 8.5, 'I': 3.6}
        self.step_up = {'g': 1, 'I': 0.1}
        self.step_down = {'g': -1, 'I': -0.1}
        self.min_step = {'g': 0.1, 'I': 0.05}
        self.max_speed = 6000 #RPM
        self.min_speed = 2000 #RPM

        #init all controllers
        self.ff_controller = feed_forward(SS_model, param)

    
    def sub(self, dict1, dict2):

        new_dict ={}

        for v in dict1.keys():
            if v in dict2.keys():
                new_dict[v] = dict1[v] - dict2[v]

        return new_dict

    def saturate(self, out, lower, upper):

        for y in out.keys():
            if out[y] < lower[y]:
                out[y] = lower[y]
            if out[y] > upper[y]:
                out[y] = upper[y]
        
        return out

    def check_constrains(self, x_next):

        pass_constrain = True

        out = self.SS_model.predict(x_next)

        if out['Speed (RPM)'] > self.max_speed or out['Speed (RPM)'] < self.min_speed:
            pass_constrain = False

        return pass_constrain

    def compute_commend(self, x_ref, x_current):
        
        #compute controller desired commend and saturate output
        h_des = self.ff_controller.step(x_ref, x_current)
        h_des = self.saturate(h_des, self.out_min, self.out_max)

        #compute differential commend and saturate output
        h_des_diff = self.sub(h_des, x_current)
        h_des_diff = self.saturate(h_des_diff, self.step_down, self.step_up)

        #decide which order to step 
        #check step I first
        if abs(h_des_diff['I']) > self.min_step['I']:
            x_next = {
                'GV (deg)': x_current['g'],
                'torque (mNm)': self.param['torque_constant']*(x_current['I'] + h_des_diff['I'])+ self.param['torque_resistance'],
                'Flow Rate (GPM)': x_current['q']
            }
            if self.check_constrains(x_next):
                return {'I_des': x_current['I'] + h_des_diff['I']}

        #Check step g second
        if abs(h_des_diff['g']) > self.min_step['g']:
            x_next = {
                'GV (deg)': x_current['g']+ h_des_diff['g'],
                'torque (mNm)': self.param['torque_constant']*x_current['I']+ self.param['torque_resistance'],
                'Flow Rate (GPM)': x_current['q']
            }
            if self.check_constrains(x_next):
                return {'g_des': x_current['g'] + h_des_diff['g']}

        return {}

    def step(self, x_ref, x_current):
        
        if 'DP_ref' not in x_ref.keys() or 'w_ref' not in x_ref.keys():
            return {}

        h_des = self.compute_commend(x_ref, x_current)

        if self.mode == "live":
            if "I_des" in h_des.keys():
                commend = "C\n"+ str(round(h_des['I_des'],3))
                self.ser.write(commend.encode()) 
            elif "g_des" in h_des.keys():
                commend = "G\n"+ str(round(h_des['g_des'],3))
                self.ser.write(commend.encode()) 

        return h_des