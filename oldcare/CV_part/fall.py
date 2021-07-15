# -*- coding: utf-8 -*-
'''
摔倒检测模型主程序

'''

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import os
import time
import subprocess
import argparse


def check_fall_detection(frame, output_fall_path, id_card_to_name, id_card_to_type,
                         fall_time_controller, faceutil,
                         facial_expression_model, fall_model):
    TARGET_WIDTH = 64
    TARGET_HEIGHT = 64

    roi = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # determine facial expression
    (fall, normal) = fall_model.predict(roi)[0]
    label = "Fall (%.2f)" % fall if fall > normal else "Normal (%.2f)" % normal

    # display the label and bounding box rectangle on the output frame
    cv2.putText(frame, label, (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    if fall > normal:
        if fall_time_controller.fall_timing == 0:  # just start timing
            fall_time_controller.set_fall_timing(1)
            fall_time_controller.start_fall_start_time()
        else:  # alredy started timing
            fall_end_time = time.time()
            difference = fall_end_time - fall_time_controller.fall_start_time

            current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                         time.localtime(time.time()))

            if difference < fall_time_controller.fall_limit_time:
                print('[INFO] %s, 走廊, 摔倒仅出现 %.1f 秒. 忽略.' % (current_time, difference))
            else:  # strangers appear
                event_desc = '有人摔倒!!!'
                event_location = '走廊'
                print('[EVENT] %s, 走廊, 有人摔倒!!!' % (current_time))
                cv2.imwrite(os.path.join(output_fall_path,
                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), frame)  # snapshot
                # insert into database
                # command = '%s inserting.py --event_desc %s --event_type 3 --event_location %s' % (
                # python_path, event_desc, event_location)
                # p = subprocess.Popen(command, shell=True)

    return frame
