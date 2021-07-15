from imutils.video import FPS

from oldcare.track import CentroidTracker


class Fence_Tools:
    def __init__(self):
        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.W = None
        self.H = None

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0

        self.skip_frames = 30

        self.init_fps = 0
        self.fps = None

    def fps_start(self):
        # start the frames per second throughput estimator
        if self.init_fps == 0:
            self.fps = FPS().start()
            self.init_fps = 1

    def set_W_H(self, W, H):
        self.W = W
        self.H = H
