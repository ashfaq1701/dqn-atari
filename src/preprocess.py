import numpy as np
import cv2 as cv


def frame_processor(frame, shape=(84, 84)):
    frame = frame.astype(np.float32) / 255.0
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160]
    frame = cv.resize(frame, shape, interpolation=cv.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))
    return frame
