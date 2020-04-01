class dynamic_model(object):

    def __init__(self, param):

        self.param = param

    def step(self, x, x_data):

        next_x = {}
        dt = self.param['dt']

        next_x['Te'] = self.param['torque_constant']*x['I'] + self.param['torque_resistance']
        next_x['V'] = x['w'] * self.param['RPM_to_radpersec'] / self.param['speed_constant'] - self.param['internal_resistance'] * x['I']  

        next_x['t'] = x['t'] + dt
        next_x['w'] = x['w'] + 1/self.param['H'] * (x['Tm'] - next_x['Te'])*dt / self.param['RPM_to_radpersec']
        if len(x_data) >= 2:
            next_x['q'] = x['q'] - 1/self.param['Tw'] * (x['DP'] - x_data[-2][3])*dt
        else:
            next_x['q'] = x['q']

        return next_x
        
        
