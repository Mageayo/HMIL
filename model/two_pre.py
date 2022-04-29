# -*-coding:utf-8-*-
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

class lstm_processing(nn.Module):
    def __init__(self, feature_num=14, hidden_dim=512):  # hidden_dim=1024
        ''' define the LSTM regression network '''
        super(lstm_processing, self).__init__()
        self.L = 64
        self.lstm1 = nn.LSTM(46, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(5, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(24, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        # self.lstm1 = nn.LSTM(46, hidden_dim, num_layers=2, batch_first=True)
        # self.lstm = nn.LSTM(5, hidden_dim, num_layers=2, batch_first=True)
        # self.lstm2 = nn.LSTM(24, hidden_dim, num_layers=2, batch_first=True)
        self.embedding1 = torch.nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.embedding = torch.nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.embedding2 = torch.nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.attention_V1 = torch.nn.Sequential(
            nn.Linear(512,128),
            nn.Tanh()
        )
        self.attention_U1 = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights1 = nn.Linear(128,1)
        self.classify1 = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 9)
        )
        self.sub1 = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.L)
        )



        self.attention_V = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(128, 1)
        self.classify = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 9)
        )
        self.sub = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.L)
        )

        self.attention_V2 = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U2 = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights2 = nn.Linear(128, 1)
        self.classify2 = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 9)
        )
        self.sub2 = torch.nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.L)
        )

        self.label_weight = torch.nn.Parameter(torch.randn(3, 9))

        self.regression1 = torch.nn.Sequential(
            nn.Linear(9, 1),
            nn.Sigmoid()
        )



        self.face_attention_V = torch.nn.Sequential(
            nn.Linear(self.L, 16),
            nn.Tanh()
        )
        self.face_attention_U = torch.nn.Sequential(
            nn.Linear(self.L, 16),
            nn.Sigmoid()
        )
        self.face_attention_weights = nn.Linear(16, 1)
        self.face_classify = torch.nn.Sequential(
            nn.Linear(64, 9)
        )

        self.head_attention_V = torch.nn.Sequential(
            nn.Linear(self.L, 16),
            nn.Tanh()
        )
        self.head_attention_U = torch.nn.Sequential(
            nn.Linear(self.L, 16),
            nn.Sigmoid()
        )
        self.head_attention_weights = nn.Linear(16, 1)
        self.head_classify = torch.nn.Sequential(
            nn.Linear(64, 9)
        )

        self.pose_attention_V = torch.nn.Sequential(
            nn.Linear(self.L, 16),
            nn.Tanh()
        )
        self.pose_attention_U = torch.nn.Sequential(
            nn.Linear(self.L, 16),
            nn.Sigmoid()
        )
        self.pose_attention_weights = nn.Linear(16, 1)
        self.pose_classify = torch.nn.Sequential(
            nn.Linear(64, 9)
        )
        self.seg_weight = torch.nn.Parameter(torch.randn(3, 9))
        self.regression2 = torch.nn.Sequential(
             nn.Linear(9, 1),
             nn.Sigmoid()
         )
        # for k in self.parameters():
        #     print(k)
            # if k != 'label_weight' or k != 'seg_weight' or k != 'regression1.0.weight' or k != 'regression1.0.bias' \
            #         or k != 'regression2.0.weight' or k != 'regression2.0.bias':
            #     v.requires_grad = False
            #     print(k)
            #     print(v.requires_grad)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)



    def forward(self, face_input, head_input, pose_input):
        seg_num = 100
        seg_frame_num = 30
        face_class_result = list()
        head_class_result = list()
        pose_class_result = list()
        face_emd_feature = list()
        head_emd_feature = list()
        pose_emd_feature = list()
        for i in range(seg_num):
            a = face_input[:, i * seg_frame_num: (i + 1) * seg_frame_num, :]
            b = head_input[:, i * seg_frame_num: (i + 1) * seg_frame_num, :]
            c = pose_input[:, i * seg_frame_num: (i + 1) * seg_frame_num, :]

            if not hasattr(self, '_flattened'):
                self.lstm1.flatten_parameters()
                self.lstm.flatten_parameters()
                self.lstm2.flatten_parameters()
            setattr(self, '_flattened', True)

            output1, _ = self.lstm1(a)  #-1,30,1024
            output, _ = self.lstm(b)
            output2, _ = self.lstm2(c)

            output1 = output1.reshape(-1, 1024)
            output = output.reshape(-1, 1024)
            output2 = output2.reshape(-1, 1024)

            a_embedding = self.embedding1(output1)  #-1,512
            b_embedding = self.embedding(output)
            c_embedding = self.embedding2(output2)

            a_v = self.attention_V1(a_embedding)
            a_v = a_v.reshape(-1, 30, 128)
            a_u = self.attention_U1(a_embedding)
            a_u = a_u.reshape(-1, 30, 128)
            a_a = self.attention_weights1(a_v*a_u)
            a_a = torch.transpose(a_a, 2, 1)
            a_a = F.softmax(a_a, dim=2)
            a_embedding = a_embedding.reshape(-1, 30, 512)
            a_M = torch.bmm(a_a, a_embedding)
            a_fusion_feature = a_M.view(-1, 512)
            a_result = self.classify1(a_fusion_feature)
            a_final_embedding = self.sub1(a_fusion_feature)
            face_class_result.append(a_result)
            face_emd_feature.append(a_final_embedding)

            b_v = self.attention_V(b_embedding)
            b_v = b_v.reshape(-1, 30, 128)
            b_u = self.attention_U(b_embedding)
            b_u = b_u.reshape(-1, 30, 128)
            b_a = self.attention_weights(b_v * b_u)
            b_a = torch.transpose(b_a, 2, 1)
            b_a = F.softmax(b_a, dim=2)
            b_embedding = b_embedding.reshape(-1, 30, 512)
            b_M = torch.bmm(b_a, b_embedding)
            b_fusion_feature = b_M.view(-1, 512)
            b_result = self.classify(b_fusion_feature)
            b_final_embedding = self.sub(b_fusion_feature)
            head_class_result.append(b_result)
            head_emd_feature.append(b_final_embedding)

            c_v = self.attention_V2(c_embedding)
            c_v = c_v.reshape(-1, 30, 128)
            c_u = self.attention_U2(c_embedding)
            c_u = c_u.reshape(-1, 30, 128)
            c_a = self.attention_weights2(c_v * c_u)
            c_a = torch.transpose(c_a, 2, 1)
            c_a = F.softmax(c_a, dim=2)
            c_embedding = c_embedding.reshape(-1, 30, 512)
            c_M = torch.bmm(c_a, c_embedding)
            c_fusion_feature = c_M.view(-1, 512)
            c_result = self.classify2(c_fusion_feature)
            c_final_embedding = self.sub2(c_fusion_feature)
            pose_class_result.append(c_result)
            pose_emd_feature.append(c_final_embedding)

        lstm_class_result1 = torch.stack(face_class_result, dim=1)
        lstm_class_result = torch.stack(head_class_result, dim=1)
        lstm_class_result2 = torch.stack(pose_class_result, dim=1)
        lstm_emd_feature1 = torch.stack(face_emd_feature, dim=1)
        lstm_emd_feature = torch.stack(head_emd_feature, dim=1)
        lstm_emd_feature2 = torch.stack(pose_emd_feature, dim=1)

        lstm_mean_result1 = lstm_class_result1.mean(dim=1)
        lstm_mean_result = lstm_class_result.mean(dim=1)
        lstm_mean_result2 = lstm_class_result2.mean(dim=1)

        label_weight = F.softmax(self.label_weight, 0)
        final_class_output1 = torch.stack([lstm_mean_result1, lstm_mean_result, lstm_mean_result2], 1)

        #print(final_class_output1.shape)

        class_output1 = torch.mul(final_class_output1, label_weight)
        class_output1 = class_output1.sum(axis = 1)

        pred_class1 = self.regression1(class_output1)

        lstm_emd_feature1 = lstm_emd_feature1.view(-1, self.L)
        lstm_emd_feature = lstm_emd_feature.view(-1, self.L)
        lstm_emd_feature2 = lstm_emd_feature2.view(-1, self.L)

        face_v = self.face_attention_V(lstm_emd_feature1)
        face_v = face_v.reshape(-1, 100, 16)
        face_u = self.face_attention_U(lstm_emd_feature1)
        face_u = face_u.reshape(-1, 100, 16)
        face_a = self.face_attention_weights(face_v*face_u)
        face_a = torch.transpose(face_a, 2, 1)
        face_a = F.softmax(face_a, dim=2)
        lstm_emd_feature1 = lstm_emd_feature1.reshape(-1, 100, 64)
        face_M = torch.bmm(face_a, lstm_emd_feature1)
        face_fusion_feature = face_M.view(-1, 64)
        face_result = self.face_classify(face_fusion_feature)

        head_v = self.head_attention_V(lstm_emd_feature)
        head_v = head_v.reshape(-1, 100, 16)
        head_u = self.head_attention_U(lstm_emd_feature)
        head_u = head_u.reshape(-1, 100, 16)
        head_a = self.head_attention_weights(head_v * head_u)
        head_a = torch.transpose(head_a, 2, 1)
        head_a = F.softmax(head_a, dim=2)
        lstm_emd_feature = lstm_emd_feature.reshape(-1, 100, 64)
        head_M = torch.bmm(head_a, lstm_emd_feature)
        head_fusion_feature = head_M.view(-1, 64)
        head_result = self.head_classify(head_fusion_feature)

        pose_v = self.pose_attention_V(lstm_emd_feature2)
        pose_v = pose_v.reshape(-1, 100, 16)
        pose_u = self.pose_attention_U(lstm_emd_feature2)
        pose_u = pose_u.reshape(-1, 100, 16)
        pose_a = self.pose_attention_weights(pose_v * pose_u)
        pose_a = torch.transpose(pose_a, 2, 1)
        pose_a = F.softmax(pose_a, dim=2)
        lstm_emd_feature2 = lstm_emd_feature2.reshape(-1, 100, 64)
        pose_M = torch.bmm(pose_a, lstm_emd_feature2)
        pose_fusion_feature = pose_M.view(-1, 64)
        pose_result = self.pose_classify(pose_fusion_feature)

        seg_weight = F.softmax(self.seg_weight, 0)
        seg_output = torch.stack([face_result, head_result, pose_result], 1)
        class_output2 = torch.mul(seg_output, seg_weight)
        class_output2 = class_output2.sum(axis=1)

        class_output = (class_output1+class_output2)/2
        pred_class2 = self.regression2(class_output2)
        #pred_class = (pred_class1+pred_class2)/2

        return pred_class1, pred_class2