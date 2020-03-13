if __name__ == '__main__':

    #imported libraries
    import atexit
    from serial import Serial
    from pynput import keyboard

    #custom libraries
    import sensor_read
    import Turbine_model
    import Turbine_controller

    # 
    # data recorder initialization
    # 
    serial_port1 = '/dev/ttyACM0'
    baud_rate1 = 9600  # In arduino, Serial.begin(baud_rate)
    write_to_file_path = "/home/lilly/Energy_Harvester/Data"
    file_name = "/2020_03_13"

    ser_sensors = Serial(serial_port1, baud_rate1)

    sensors = sensor_read.sensor_recorder(ser_sensors, print_time=1, save_rate=200,
                                                       plot_time=5, save_path=write_to_file_path + file_name)

    # 
    # Controller initialization
    # 
    serial_port2 = '/dev/ttyACM1'
    baud_rate2 = 9600  # In arduino, Serial.begin(baud_rate)
    output_var_con = ['GV (deg)', 'I (A)']
    input_var_con = ['DP (psi)', 'Speed (RPM)', 'Flow Rate (GPM)' ]

    ser_actuator = Serial(serial_port2, baud_rate2)

    steady_state_model = Turbine_model.SS_model(load_model = True)
    controller = Turbine_controller.controller_main(ser_actuator, steady_state_model, input_var_con, output_var_con)

    #
    #save data on exit
    #
    atexit.register(sensors.save_data, exit=1)

    #
    #set up keyboard interaction (non-blocking)
    #
    listener = keyboard.Listener(on_press=sensors.user_input,
                                 on_release=controller.user_input)
    listener.start()

    print("\n##########################\nStart reading!\n##########################\n")

    while True:

        if sensors.record_:
            sensors.read()
            sensors.print_to_screen()
            sensors.average_steady_state_data()

            controller.update_current_state(sensors.current_states())

    