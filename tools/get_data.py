#jia zai shu ju ji
#load dataset
import os, sys, shutil
import random as rd
from os import listdir
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.utils.data as data

def load_imgs(image_label_path_list):
    label = list()
    all_data = list()
    feature_length = 3000
    for image_label_path in image_label_path_list:
        with open(image_label_path, "r") as file:
            for id, label_line in enumerate(file):
                file, label = label_line.strip().split(" ")
                file_name = file.split('.')[0]

                num = int(file_name)
                # if num >= 5:
                #     break
                #print(file_name)
               # gather_frame_path = "/home/mjy/data/gather_result/mean_3000/"
                #gather_frame_path = "/home/mjy/data/gather_result/mean_3000_fit/"
                #gather_frame_path = "/home/mjy/data/gather_result/mean_3000_fit_head/"
                #gather_frame_path = "/home/mjy/data/gather_result/mean_fit_3000/"

                #gather_frame_path = "/home/mjy/data/gather_result/mean_center_3000_head/"
                #gather_frame_path = "/home/mjy/data/gather_result/knnmean_3000/"


                gather_frame_path = "/data/mjy/OriginalData/gather_result/mean_center_3000/"

                #old_mooc_test
                #gather_frame_path = "/home/mjy/data/gather_result/original_fit_norm_mean_3000_1/"
                #gather_frame_path = "/home/mjy/data/work_train/mean_center_3000/"

                #gather_frame_path = "/home/mjy/data/gather_result/random_norm_mean_3000/"
                #gather_frame_path = "/home/mjy/data/work_train/random_norm_mean_3000/"

                face_file = gather_frame_path + "face/" + file_name + ".txt"
                pose_file = gather_frame_path + "pose/" + file_name + ".txt"
                head_file = gather_frame_path + "head/" + file_name + ".txt"
                with open(face_file, 'r') as f:

                    pose_feature = list()
                    face_feature = list()
                    head_feature = list()
                    for index, line in enumerate(f):
                        if index >= feature_length:
                            break
                        face_f = line.strip().split(' ')
                        # if len(face_f) != 196:
                        #     print(index, face_file, len(face_f))
                        face_feature.append(face_f)
                face_feature_numpy = np.array(face_feature)
                face_feature_numpy = face_feature_numpy.astype(float)

                with open(pose_file, 'r') as f1:
                    for index, line in enumerate(f1):
                        if index >= feature_length:
                            break
                        pose_f = line.strip().split(' ')
                        #print(pose_f)
                        # if len(pose_f) != 34:
                        #     print(index, pose_file, len(pose_f))
                        pose_feature.append(pose_f)
                        #break
                pose_feature_numpy = np.array(pose_feature)
                pose_feature_numpy = pose_feature_numpy.astype(float)
                with open(head_file, 'r') as f2:
                    for index, line in enumerate(f2):
                        if index >= feature_length:
                            break
                        head_f = line.strip().split(' ')
                        # if len(pose_f) != 34:
                        #     print(index, pose_file, len(pose_f))
                        head_feature.append(head_f)
                        #break
                head_feature_numpy = np.array(head_feature)
                head_feature_numpy = head_feature_numpy.astype(float)

                label = float(int(label))/8
                if float(label) == 0.0:
                    classification_label = int(0)
                if float(label) == 0.125:
                    classification_label = int(1)
                if float(label) == 0.25:
                    classification_label = int(2)
                if float(label) == 0.375:
                    classification_label = int(3)
                if float(label) == 0.5:
                    classification_label = int(4)
                if float(label) == 0.625:
                    classification_label = int(5)
                if float(label) == 0.75:
                    classification_label = int(6)
                if float(label) == 0.875:
                    classification_label = int(7)
                if float(label) == 1.0:
                    classification_label = int(8)
                #print(label)
                all_data.append((face_feature_numpy, head_feature_numpy, pose_feature_numpy, label, classification_label))
            break
    #print(all_data[0][0].shape)
    return all_data


class Dataset(data.Dataset):
	def __init__(self, image_list_file):
		self.all_datas = load_imgs(image_list_file)

	def __getitem__(self, index):
		face_feature_numpy, head_feature_numpy, pose_feature_numpy, label, classification_label = self.all_datas[index]
		return face_feature_numpy, head_feature_numpy, pose_feature_numpy, label, classification_label

	def __len__(self):
		return len(self.all_datas)

if __name__ == '__main__':
    all_datas = load_imgs(["./label_file.txt"])
    all_datas_numpy = np.array(all_datas)

    for i in range(5):
        print(all_datas_numpy[i])
        # print(all_datas_numpy[i][0].shape)
        # print(all_datas_numpy[i][1].shape)
        # print(all_datas_numpy[i][2].shape)

# import os, sys, shutil
# import random as rd
# from os import listdir
# from PIL import Image
# import numpy as np
# import random
# import torch
# import torch.nn.functional as F
# import torch.utils.data as data
# #sys.path.append("/data/mjy/code/engage220308/")
# from tools.feature_process import process_face_feature, process_pose_feature, process_head_feature
# def load_imgs(image_label_path_list):
#     label = list()
#     all_data = list()
#     feature_length = 300
#     for image_label_path in image_label_path_list:
#         with open(image_label_path, "r") as file:
#             for id, label_line in enumerate(file):
#                 file, label = label_line.strip().split(" ")
#                 file_name = file.split('/')[1].split('.')[0]

#                 gather_frame_path = "/data/mjy/OriginalData/gather_result/mean_center_3000/"

#                 face_file = gather_frame_path + "face/" + file_name + ".txt"
#                 pose_file = gather_frame_path + "pose/" + file_name + ".txt"
#                 head_file = gather_frame_path + "head/" + file_name + ".txt"

#                 with open(face_file, 'r') as f:

#                     pose_feature = list()
#                     face_feature = list()
#                     head_feature = list()
#                     for index, line in enumerate(f):
#                         if index >= feature_length:
#                             break
#                         if index % 10 == 0:
#                             face_f = line.strip().split(' ')
#                             face_f = np.array(face_f).astype(float)
#                             # if len(face_f) != 196:
#                             #     print(index, face_file, len(face_f))
#                             face_f = process_face_feature(face_f)
# #                             if face_f.shape[0]!=46:
# #                                 print(face_file)
                            
#                             face_feature.append(face_f)
#                 face_feature_numpy = np.array(face_feature)
#                 #print(face_feature_numpy)
#                 face_feature_numpy = face_feature_numpy.astype(float)
#                 if face_feature_numpy.shape[0] != 30:
#                     print("face_file is wrong in",face_file)
#                 with open(pose_file, 'r') as f1:
#                     for index, line in enumerate(f1):
#                         if index >= feature_length:
#                             break
#                         if index % 10 == 0:
#                             pose_f = line.strip().split(' ')
#                             pose_f = np.array(pose_f).astype(float)
#                             # if len(face_f) != 196:
#                             #     print(index, face_file, len(face_f))
#                             pose_f = process_pose_feature(pose_f)
#                             pose_feature.append(pose_f)
#                 pose_feature_numpy = np.array(pose_feature)
#                 pose_feature_numpy = pose_feature_numpy.astype(float)
#                 if pose_feature_numpy.shape[0] != 30:
#                     print("pose_file is wrong in", pose_file)

#                 with open(head_file, 'r') as f2:
#                     for index, line in enumerate(f2):
#                         if index >= feature_length:
#                             break
#                         if index % 10 == 0:
#                             head_f = line.strip().split(' ')
#                             head_f = np.array(head_f).astype(float)
#                             # if len(face_f) != 196:
#                             #     print(index, face_file, len(face_f))
#                             head_f = process_head_feature(head_f)
#                             head_feature.append(head_f)
#                 head_feature_numpy = np.array(head_feature)
#                 head_feature_numpy = head_feature_numpy.astype(float)
#                 if head_feature_numpy.shape[0] != 30:
#                     print("head_file is wrong in", head_file)
#                 #print(head_feature_numpy)
#                 label = float(int(label))/8
#                 if float(label) == 0.0:
#                     classification_label = int(0)
#                 if float(label) == 0.125:
#                     classification_label = int(1)
#                 if float(label) == 0.25:
#                     classification_label = int(2)
#                 if float(label) == 0.375:
#                     classification_label = int(3)
#                 if float(label) == 0.5:
#                     classification_label = int(4)
#                 if float(label) == 0.625:
#                     classification_label = int(5)
#                 if float(label) == 0.75:
#                     classification_label = int(6)
#                 if float(label) == 0.875:
#                     classification_label = int(7)
#                 if float(label) == 1.0:
#                     classification_label = int(8)
#                 #print(label)
#                 all_data.append((face_feature_numpy, head_feature_numpy, pose_feature_numpy, label, classification_label))
#             #break
#     #print(all_data[0][0].shape)
#     return all_data


# class Dataset(data.Dataset):
# 	def __init__(self, image_list_file):
# 		self.all_datas = load_imgs(image_list_file)

# 	def __getitem__(self, index):
# 		face_feature_numpy, head_feature_numpy, pose_feature_numpy, label, classification_label = self.all_datas[index]
# 		return face_feature_numpy, head_feature_numpy, pose_feature_numpy, label, classification_label

# 	def __len__(self):
# 		return len(self.all_datas)

# if __name__ == '__main__':
#     all_datas = load_imgs(["/data/mjy/code/engage220308/labels/0309/train_0309.txt"])
#     all_datas_numpy = np.array(all_datas)

#     # for i in range(5):
#     #     print(all_datas_numpy[i])
#         # print(all_datas_numpy[i][0].shape)
#         # print(all_datas_numpy[i][1].shape)
#         # print(all_datas_numpy[i][2].shape)
