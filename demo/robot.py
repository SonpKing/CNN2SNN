from .l_short_comm import Get
import time
class Robot:
    def __init__(self, IP="192.168.2.100", Port=13001):
        self.last_robot_cmd = -1
        self.last_rotate_time = time.time()
        self.last_robot_time = time.time()
        self.robot_IP = IP
        self.robot_Port = Port

    def control(self, is_water, is_house, is_broker, is_person):
        print(is_water, is_house, is_broker, is_person)
        try:
            cur_time = time.time()
            self.check_cmd(cur_time)
            if self.need_repair(is_water):
                self.get_repair(cur_time)
            elif self.need_medical(is_person):
                self.get_medical()
            elif self.need_rotate(is_house, is_broker, cur_time):
                self.get_rotate()
        except:
            pass

    def check_cmd(self, cur_time):
        if cur_time - self.last_robot_time > 10:
            self.last_robot_cmd = -1

    def need_medical(self, is_person):
        return is_person and self.last_robot_cmd != 0
        
    def need_repair(self, is_water):
        return is_water and self.last_robot_cmd != 1

    def need_rotate(self, is_house, is_broker, cur_time):
        if cur_time - self.last_rotate_time < 30:
            can_rotate = False
        else:
            can_rotate = True
        return (is_house or is_broker) and self.last_robot_cmd != 2 and can_rotate

    def get_medical(self):
        Get(self.robot_IP, self.robot_Port, 100, bytearray(2))
        # Get(self.robot_IP, self.robot_Port, 104, bytearray(2))
        print("!!!!!!!!!!!!!!!send person")
        self.last_robot_cmd = 0

    def get_repair(self, cur_time):
        Get(self.robot_IP, self.robot_Port, 102, bytearray(2))
        # Get(self.robot_IP, self.robot_Port, 103, bytearray(2))
        Get(self.robot_IP, self.robot_Port, 104, bytearray(2))
        print("!!!!!!!!!!!!!!!send broker")
        self.last_robot_cmd = 1
        self.last_rotate_time = cur_time

    def get_rotate(self):
        self.last_robot_cmd = 2
        pass
