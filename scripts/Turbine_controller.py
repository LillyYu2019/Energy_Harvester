import Turbine_model
from datetime import datetime

class feed_forward(object):

    def __init__(self, SS_model):

        super().__init__()

        self.SS_model = SS_model

    def predict(self, x_desired):

        if 'I (A)' in x_desired.keys():
            x_desired['torque (mNm)'] = self.SS_model.current_to_torque(x_desired['I (A)'])
            del x_desired['I (A)']

        if 'Speed (rpm)' in x_desired.keys():
            x_desired['Speed (rad/s)'] = self.SS_model.RPM_to_rads(x_desired['Speed (rpm)'])
            del x_desired['Speed (rpm)']
        print(x_desired)
        commend = self.SS_model.predict(x_desired)
        print(commend)
        if 'torque (mNm)' in commend.keys():
            commend['I (A)'] = self.SS_model.torque_to_current(commend['torque (mNm)'])
            del commend['torque (mNm)']

        if 'Speed (rad/s)' in commend.keys():
            commend['Speed (rpm)'] = self.SS_model.rads_to_RPM(commend['Speed (rad/s)'])
            del commend['Speed (rad/s)']

        return commend

class controller_main(object):

    def __init__(self, SS_model, input_var, output_var):

        self.input_var = input_var
        self.output_var = output_var

        self.ff_controller = feed_forward(SS_model)

        self.x_desired = {'DP (psi)':19.9, 'Speed (rad/s)':567, 'Flow Rate (GPM)':25.82}
        self.x_current = {'DP (psi)':13.6, 'Speed (rad/s)':505, 'Flow Rate (GPM)':25.82}
        self.y_desired = {}
        self.y_current = {'GV (deg)':6.5, 'I (A)':2.22}

        #flags
        self.new_commend_computed = False
        self.reading_data = True

        #constants
        self.out_min = {'GV (deg)': 0.0, 'I (A)': 0.8}
        self.out_max = {'GV (deg)': 8.2, 'I (A)': 3.2}
        self.step_size = {'GV (deg)': 1, 'I (A)': 0.4}
        self.max_speed = 4500 #RPM

        self.current_time = datetime.now()
        self.previous_time = self.current_time

    def saturate(self, out):

        for y in out.keys():
            if out[y] < self.out_min[y]:
                out[y] = self.out_min[y]
            if out[y] > self.out_min[y]:
                out[y] = self.out_max[y]
        
        return out

    def get_commend(self):

        self.y_desired = self.saturate(self.ff_controller.predict(self.x_desired))
        self.new_commend_computed = True

    def step_I(self, I):
        #sends a bunch of commends to arduino
        print("output step I: " + str(I))
        return

    def step_GV(self, GV):
        #sends a bunch of commends to arduino
        print("output step GV: " + str(GV))
        return

    def step(self):

        if not self.reading_data or not self.new_commend_computed:
            return
        
        diff = {}

        for y in self.y_desired.keys():
            diff[y] = self.y_desired[y] - self.y_current[y]
            if diff[y] > self.step_size[y]:
                diff[y] = self.step_size[y]
            if diff[y] < -self.step_size[y]:
                diff[y] = -self.step_size[y]

        print(self.y_desired)
        print(diff)

        #check if step current first is okay:
        y_next = self.ff_controller.predict({'GV (deg)': self.y_current['GV (deg)'], 
                                             'I (A)': self.y_current['I (A)'] + diff['I (A)'],
                                             'Flow Rate (GPM)': self.x_current['Flow Rate (GPM)']})
        print(y_next)
        if y_next['Speed (rpm)'] < self.max_speed:
            self.step_I(diff['I (A)'])
            self.step_GV(diff['GV (deg)'])
            self.new_commend_computed = False
            return True

        #check if step Gv angle first is okay:
        y_next = self.ff_controller.predict({'GV (deg)': self.y_current['GV (deg)'] + diff['GV (deg)'], 
                                             'I (A)': self.y_current['I (A)'],
                                             'Flow Rate (GPM)': self.x_current['Flow Rate (GPM)']})
        print(y_next)
        if y_next['Speed (rpm)'] < self.max_speed:
            self.step_GV(diff['GV (deg)'])
            self.step_I(diff['I (A)'])
            self.new_commend_computed = False
            return True

        return False

        





if __name__ == '__main__':

    #controller
    output_var_con = ['GV (deg)', 'I (A)']
    input_var_con = ['DP (psi)', 'Speed (RPM)', 'Flow Rate (GPM)' ]

    steady_state_model = Turbine_model.SS_model(load_model = True)
    controller = controller_main(steady_state_model, input_var_con, output_var_con)
    controller.get_commend()
    controller.step()