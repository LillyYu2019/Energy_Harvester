import numpy as np

class dynamic_model(object):

    def __init__(self, SS_model, Tw = 0.1 , H = 0.8)):

        #Turbine model
        self.SS_model = SS_model

        #model constants:
        self.Tw = Tw
        self.H = H
        self.torque_constant = 46.541 #mNm/A
        self.torque_resistance = 10.506 #mNm
        self.speed_constant = 216.943 * 0.104719755 #rad/s/V
        self.internal_resistance = 2.434
        self.minor_loss = 0.491 #psi
        self.density_water = 997 #kg/m^3
        self.m3_to_GPM = 15.89806024
        self.psi_to_Pa = 6894.76

        self.variables = ['DP', 
                          'w', 
                          'q', 
                          'g', 
                          'Tm',
                          'I'
                          'Te',
                          'V',
                          'P_f',
                          'P_t',
                          'P_e',
                          'eff_t',
                          'eff_g']
        self.state = {}

    def set_init_states(self, init)
        
        for i, var in enumerate(self.variables[0:6]):
            self.state[var] = init[self.SS_model.variables[i]]

        self.update_all()

    def update_all(self):

        self.state['Te'] = self.torque_constant*self.state['I'] + self.torque_resistance 
        self.state['V'] = self.state['w'] / self.speed_constant - self.internal_resistance * self.state['I']  
        self.state['P_e'] = self.state['V'] * self.state['I']
        self.state['P_t'] = self.state['Tm'] * self.state['w'] / 1000.0
        self.state['P_f'] = self.state['q'] / self.m3_to_GPM * (self.state['DP'] - self.minor_loss)*self.psi_to_Pa/self.density_water
        self.state['eff_t'] = self.state['P_t'] / self.state['P_f'] 
        self.state['eff_g'] = self.


    def intergrate_w(self, dt):
        return self.x['torque (mNm)'] + 1/self.H * (self.current_state.Tm - self.current_state.Te)*dt
    
    def intergrate_q(self, dt):
        return self.current_state.q - 1/self.Tw * (self.current_state.DP - self.state_t[-2][4])*dt