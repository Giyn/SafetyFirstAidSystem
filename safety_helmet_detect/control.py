import serial
import threading
import time
import struct

class Control(object):
    def __init__(self, com='COM7', baudrate=115200):
        self.ser = serial.Serial(com, baudrate)
        time.sleep(0.2)

        # thread = threading.Thread(target=self.input_cmd, args=(com,))
        # thread.setDaemon(True)
        # thread.start()

    def input_cmd(self, com):
        while True:
            cmd = input('>>> ')
            cmd_data = str(float(cmd[1:])).encode()
            if cmd[0] == 'p' or 'P':
                prefix = b'p'
            elif cmd[0] == 'i' or 'I':
                prefix = b'i'
            elif cmd[0] == 'd' or 'D':
                prefix = b'd'

            if cmd[0] == 't' or 'T':
                prefix += b't'
            elif cmd[0] == 'u' or 'U':
                prefix += b'u'

            com.write(prefix + struct.pack('B', len(cmd_data)) + cmd_data)

    def update(self, top, under):
        top_data = str(240 - top).encode()
        under_data = str(320 - under).encode()
        send_data = b't' + struct.pack('B', len(top_data)) + top_data + \
                    b'u' + struct.pack('B', len(under_data)) + under_data
        self.ser.write(send_data)

    def close(self):
        self.ser.close()
