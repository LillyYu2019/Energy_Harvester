class actuators(object):

    def __init__(self, param):

        self.param = param
        self.g_speed = 1.58 #deg/sec

    def step(self, h_des, x_current):
        
        if ('g_des' not in h_des.keys()) and ('I_des' not in h_des.keys()):
            return {}

        if 'g_des' in h_des.keys():
            diff = h_des['g_des'] > x_current['g']
            if diff > 0:
                return {'g': x_current['g'] + min(diff, self.g_speed*self.param['dt'])}
            else:
                return {'g': x_current['g'] + max(diff, -self.g_speed*self.param['dt'])}

        if 'I_des' in h_des.keys():
            return {'I': h_des['I_des']}
