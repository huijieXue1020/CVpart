import time


class Time_Controller:
    def __init__(self):
        self.strangers_timing = 0  # 计时开始
        self.strangers_start_time = 0  # 开始时间
        self.strangers_limit_time = 2  # if >= 2 seconds, then he/she is a stranger.

        # 控制微笑检测
        self.facial_expression_timing = 0  # 计时开始
        self.facial_expression_start_time = 0  # 开始时间
        self.facial_expression_limit_time = 2  # if >= 2 seconds, he/she is smiling

        # 控制陌生人检测
        self.fall_timing = 0  # 计时开始
        self.fall_start_time = 0  # 开始时间
        self.fall_limit_time = 2  # if >= 1 seconds, then he/she falls.

    def set_stranger_timing(self, strangers_timing):
        self.strangers_timing = strangers_timing

    def set_facial_expression_timing(self, facial_expression_timing):
        self.facial_expression_timing = facial_expression_timing

    def set_fall_timing(self, fall_timing):
        self.fall_timing = fall_timing

    def start_stranger_time(self):
        self.strangers_start_time = time.time()

    def start_facial_expression__time(self):
        self.facial_expression_start_time = time.time()

    def start_fall_start_time(self):
        self.fall_start_time = time.time()
