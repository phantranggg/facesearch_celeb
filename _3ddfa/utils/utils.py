from scipy.spatial import distance as dist
from imutils import face_utils
import cv2

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def ear_distance(shape):

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0

    return ear

def left_ear__distance(shape):
    leftEye = shape[lStart:lEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    return leftEAR
def right_ear_distance(shape):
    rightEye = shape[rStart:rEnd]
    rightEAR = eye_aspect_ratio(rightEye)
    return rightEAR

def brow_eye_distance(shape):
    right_distance = dist.euclidean(shape[19], shape[37])
    left_distance = dist.euclidean(shape[23], shape[43])
    return (right_distance+left_distance)/2

def smile_distance(shape):
    return dist.euclidean(shape[62], shape[66])

def node_pixel(shape):
    return shape