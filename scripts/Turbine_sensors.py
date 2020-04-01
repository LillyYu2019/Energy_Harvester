
from serial import Serial


class sensor_recorder(object):

    def __init__(self, input_port, param):

        self.ser = Serial(input_port, 9600)
        self.param = param

        self.identifier = ['t', 'PT1', 'PT2', 'Tm', 'V', 'I', 'w', 'q', 'g']
        self.x = {}
        self.invalid_data = True

    def read(self):
        '''
        Read serial port data line by line and save in internal variable
        '''
        self.invalid_data = False
        for i in self.identifier:
            read_count = 0

            try:
                line = self.ser.readline().decode()

                line_arr = line.split()

                if len(line_arr) < 1:
                    self.invalid_data = True
                    continue

                while line_arr[0] != i and read_count < 5:
                    read_count += 1
                    line = self.ser.readline().decode("utf-8")
                    line_arr = line.split()

                if read_count < 5:
                    val = float(line_arr[-1])
                    self.x[i] = val
                else:
                    print("Warning - read count > 5, read this: " + line)
                    self.invalid_data = True

            except:
                self.invalid_data = True
                continue

        self.x['DP'] = self.x['PT1'] - self.x['PT2']
        self.x['Te'] = self.param['torque_constant'] * \
            self.x['I'] + self.param['torque_resistance']

        return self.x
