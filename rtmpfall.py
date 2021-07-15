# -*- coding: utf-8 -*-

"""
测试rtmp推流

用法:


"""


import time
import cv2
import queue
from threading import Thread
import datetime, _thread
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.models import load_model
import subprocess as sp

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ANGLE = 20
# 全局常量

limit_time = 2
# 使用线程锁，防止线程死锁
mutex = _thread.allocate_lock()
# 存图片的队列
frame_queue = queue.Queue()

# rtmpUrl = "rtmp://127.0.0.1:1935/rtmplive"
rtmpUrl = "rtmp://192.168.140.109:1935/rtmplive"

command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(640, 480),  # 图片分辨率
           '-r', str(30.0),  # 视频帧率
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmpUrl]



def Video():
    # 调用相机拍图的函数
    vid = cv2.VideoCapture(0)
    time.sleep(2);
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    while (vid.isOpened()):
        return_value, frame = vid.read()

        # 原始图片推入队列中
        frame_queue.put(frame)
        frame_queue.get() if frame_queue.qsize() > 1 else time.sleep(0.01)


def push_frame():
    # 推流函数
    fall_model_path = 'models/fall_detection_googlenet_20.hdf5'

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    # prev_time = time()
    # 防止多线程时 command 未被设置
    while True:
        print('command lenth', len(command))
        if len(command) > 0:
            # 管道配置，其中用到管道
            p = sp.Popen(command, stdin=sp.PIPE)
            break

    while True:
        fall_model = load_model(fall_model_path, compile=False)
        if frame_queue.empty() != True:
            counter = 0

            while True:
                counter += 1
                image = frame_queue.get()
                # if counter%2 != 0:  # 放弃前10帧
                #     continue
                image = cv2.flip(image, 1)  #镜像翻转

                # image = Fall.check_fall_detection(image, )
                roi = cv2.resize(image, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # determine facial expression
                (fall, normal) = fall_model.predict(roi)[0]
                label = "Fall (%.2f)" % fall if fall > normal else "Normal (%.2f)" % normal
                print(label)

                cv2.putText(image, label, (image.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                image = cv2.resize(image, (640, 480))   #压缩
                p.stdin.write(image.tostring())


def run():
    # 使用两个线程处理

    thread1 = Thread(target=Video, )
    thread1.start()
    time.sleep(2)
    thread2 = Thread(target=push_frame, )
    thread2.start()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()





