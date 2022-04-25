#feature process
#te zheng chu li
import numpy as np

def process_face_feature(face_feature):
    mean_f = np.array([320, 240] * 23)
    if np.min(face_feature) < 0 and np.max(face_feature) < 0:
        face_points = np.zeros(46)
    else:
        point_index = [0, 1, 8, 9, 16, 17, 24, 25, 32, 33, 33, 34, 35, 36, 37, 38, 42, 43, 44, 45, 46, 47, 51, 52, 54,
                       55, 60, 61, 96, 97, 64, 65, 68, 69, 97, 98, 72, 73, 76, 77, 79, 80, 82, 83, 85, 86]
        face_points = face_feature[point_index]
    face_points = (face_points - mean_f)/mean_f
    #print(face_points)
    return face_points



def process_pose_feature(pose_feature):
    mean_f = np.array([320, 240] * 12)
    if np.min(pose_feature) < 0 and np.max(pose_feature) < 0:
        pose_points = np.zeros(24)
    else:
        point_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28, 29, 30, 31, 32, 33, 34, 35]
        pose_points = pose_feature[point_index]
    pose_points = (pose_points - mean_f) / mean_f
    return pose_points


def process_head_feature(head_feature):
    mean_f = np.array([0, 0, 0, 320, 240])
    range_f = np.array([90, 90, 90, 320, 240])
    if np.min(head_feature) < 0 and np.max(head_feature) < 0:
        head_points = np.zeros(5)
    else:
        head_points = head_feature
    head_points = (head_points - mean_f) / range_f
    return head_points