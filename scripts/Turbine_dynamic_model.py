class dynamic_model(object):

    def __init__(self, param):

        self.param = param
        
    def step(self, x):

        dt = self.param['dt']

        next_x = self.conduit_dynamics(x)

        next_x['Te'] = self.param['torque_constant']*x['I'] + self.param['torque_resistance']
        next_x['V'] = x['w'] / self.param['speed_constant'] - self.param['internal_resistance'] * x['I'] - self.param['V_intercept']
        next_x['w'] = x['w'] + 1/self.param['H'] * (x['Tm'] - x['Te'])*dt / self.param['RPM_to_radpersec']
        next_x['q'] = x['q'] + 1/self.param['Tw'] * (x['PT1'] - x['DP'] - x['PT2'])*dt
        
        return next_x

    def conduit_dynamics(self, x):

        coe = self.param['PT1_constants'][self.param['pump_level']]
        
        next_x = {}
        next_x['PT1'] = 0
        for i, c in enumerate(coe):
            next_x['PT1'] += c*x['q']**i #psi

        next_x['PT2'] = 0
        for i, c in enumerate(self.param['PT2_constants']):
            next_x['PT2'] += c*x['q']**i #psi

        return next_x