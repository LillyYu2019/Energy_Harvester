class actuators(object):

    def __init__(self, param):

        self.param = param
        self.g_speed = 1 #deg/sec
        self.I_speed = 0.6 / 1 #A/sec
        self.g_des = 0
        self.I_des = 0
        self.state = 3
        self.last_state = 2

    def step(self, h_des, x_current):
        '''

        state 0 - new command received, assess which to move first
        state 1 - move g
        state 2 - move I
        state 3 - controller paused

        '''
        if 'g_des' in h_des.keys() and 'I_des' in h_des.keys():
            self.g_des = h_des['g_des']
            self.I_des = h_des['I_des']
            self.last_state = self.state
            self.state = 0
        
        if self.state == 0:
            if self.g_des >= x_current['g'] and self.I_des >= x_current['I_measured']:
                self.state = 2
            else:
                
                if self.last_state == 2:
                    self.state = 1
                else:
                    self.state = 2


        if self.state == 1 :
            diff = self.g_des - x_current['g']
            if abs(diff) < 0.005:
                self.state = 2
            else:
                if diff > 0:
                    return {'g': x_current['g'] + min(diff, self.g_speed*self.param['dt'])}
                else:
                    return {'g': x_current['g'] + max(diff, -self.g_speed*self.param['dt'])}

        if self.state == 2:
            diff = self.I_des - x_current['I']
            if abs(diff) < 0.005:
                self.state = 1
            else:
                if diff > 0:
                    return {'I': x_current['I'] + min(diff, self.I_speed*self.param['dt'])}
                else:
                    return {'I': x_current['I'] + max(diff, -self.I_speed*self.param['dt'])}

        return {}
