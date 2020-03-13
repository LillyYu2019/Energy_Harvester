from datetime import datetime

class feed_forward(object):

    def __init__(self, SS_model):

        super().__init__()

        self.SS_model = SS_model

    def predict(self, x_desired):

        if 'I (A)' in x_desired.keys():
            x_desired['torque (mNm)'] = self.SS_model.current_to_torque(x_desired['I (A)'])

        if 'Speed (rpm)' in x_desired.keys():
            x_desired['Speed (rad/s)'] = self.SS_model.RPM_to_rads(x_desired['Speed (rpm)'])

        commend = self.SS_model.predict(x_desired)

        if 'torque (mNm)' in commend.keys():
            commend['I (A)'] = self.SS_model.torque_to_current(commend['torque (mNm)'])

        if 'Speed (rad/s)' in commend.keys():
            commend['Speed (rpm)'] = self.SS_model.rads_to_RPM(commend['Speed (rad/s)'])

        return commend

class controller_main(object):

    def __init__(self, ser, SS_model, input_var, output_var):

        self.ser = ser
        self.input_var = input_var
        self.output_var = output_var

        self.ff_controller = feed_forward(SS_model)

        self.x_desired = {'DP (psi)':11.3, 'Speed (rpm)':4267, 'Flow Rate (GPM)':25.0}
        self.x_current = {'DP (psi)':18.76, 'Speed (rpm)':4549, 'Flow Rate (GPM)':25.0}
        self.y_desired = {}
        self.y_current = {'GV (deg)':8.5, 'I (A)':3.4}
        self.diff = {}
        self.commend = ""

        #flags
        self.new_commend_computed = False
        self.reading_data = True

        #constants
        self.out_min = {'GV (deg)': 0.0, 'I (A)': 0.8}
        self.out_max = {'GV (deg)': 8.5, 'I (A)': 3.6}
        self.step_up = {'GV (deg)': 1, 'I (A)': 0.4}
        self.step_down = {'GV (deg)': -1, 'I (A)': -0.4}
        self.max_speed = 5000 #RPM

        self.current_time = datetime.now()
        self.previous_time = self.current_time

    def add(self, dict1, dict2):

        new_dict ={}

        for v in dict1.keys():
            if v in dict2.keys():
                new_dict[v] = dict1[v] + dict2[v]

        return new_dict

    def sub(self, dict1, dict2):

        new_dict ={}

        for v in dict1.keys():
            if v in dict2.keys():
                new_dict[v] = dict1[v] - dict2[v]

        return new_dict

    def saturate(self, out, out_min, out_max):

        for y in self.output_var:
            if out[y] < out_min[y]:
                out[y] = out_min[y]
            if out[y] > out_max[y]:
                out[y] = out_max[y]
        
        return out

    def update_current_state(self, sensor_data):

        if sensor_data == False:
            return 

        self.x_current = {}
        self.y_current = {}

        for var in self.input_var:
            self.x_current[var] = sensor_data[var]
        
        for var in self.output_var:
            self.y_current[var] = sensor_data[var]

        self.reading_data = True

    def get_commend(self):

        self.y_desired = self.saturate(self.ff_controller.predict(self.x_desired), self.out_min, self.out_max)
        self.new_commend_computed = True

    def step_I(self):
        #sends a bunch of commends to arduino
        print("output step I: " + str(self.diff['I (A)']))
        self.commend = "C\n"+ str(round(self.diff['I (A)'],3))
        return

    def step_GV(self):
        #sends a bunch of commends to arduino
        print("output step GV: " + str(self.diff['GV (deg)']))
        self.commend = "G\n"+ str(round(self.diff['GV (deg)'],3))
        return

    def predict_x_next(self, var=[]):

        new_diff = self.diff.copy()
        for v in new_diff.keys():
            if var and v not in var:
                new_diff[v] = 0.0

        pred = {'GV (deg)': self.y_current['GV (deg)'] + new_diff['GV (deg)'], 
                                             'I (A)': self.y_current['I (A)'] + new_diff['I (A)'],
                                             'Flow Rate (GPM)': self.x_current['Flow Rate (GPM)']}

        x_next = self.ff_controller.predict(pred)
        return x_next

    def step(self):

        if not self.reading_data:
            return

        controller.get_commend()

        self.diff = self.sub(self.y_desired, self.y_current)
        self.diff = self.saturate(self.diff, self.step_down, self.step_up)

        can_step_I = False
        can_step_GV = False

        if self.predict_x_next()['Speed (rpm)'] >= self.max_speed:
            print("Desired x not possible, max speed exceeded")
            return False
        
        if abs(self.diff['I (A)']) > 0.1 and self.predict_x_next(var=['I (A)'])['Speed (rpm)'] < self.max_speed:
            can_step_I = True

        if abs(self.diff['GV (deg)']) > 0.2 and self.predict_x_next(var=['GV (deg)'])['Speed (rpm)'] < self.max_speed:
            can_step_GV = True

        if can_step_I:
            self.step_I()
            return True

        if can_step_GV:
            self.step_GV()
            return True

        return False

    def run(self, max_iteration = 10):

        for i in range(max_iteration):
            controller.step()

    def user_input(self, key):

        try:
            s = key.char
        except:
            s = ""

        if s == 'c':
            #controller.step()
            self.diff['GV (deg)'] = 1
            self.step_GV()
            self.ser.write(self.commend).encode("utf-8")

        return True


if __name__ == '__main__':

    import Turbine_model

    #controller
    output_var_con = ['GV (deg)', 'I (A)']
    input_var_con = ['DP (psi)', 'Speed (RPM)', 'Flow Rate (GPM)' ]

    steady_state_model = Turbine_model.SS_model(load_model = True)
    controller = controller_main(steady_state_model, input_var_con, output_var_con)
    controller.run()
    