import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import itertools
import pandas as pd
import numpy as np
from six.moves import cPickle as pickle
import sys
from sklearn.metrics import confusion_matrix
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from textwrap import wrap
print(torch.__version__)
print(torch.cuda.is_available())
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from tqdm.notebook import tqdm # 프로세스 바

device = torch.device("cuda:%d" % 0 if torch.cuda.is_available() else "cpu")
Imgshape = (224,224)
channel = 3
l2norm = 0.001; drop_prob = 0.5
ratio = 0.2;patience=100;
num_epochs =2000;
batch_size = 64;
learning_rate = 1e-4;

# from torch.utils.tensorboard import SummaryWriter
datapath = '\\\\147.47.239.143\\SHRM-Personal\\Personal_Drive\\박찬희\\RESEARCH_2022\\1. PYCODE\\Motor_Signal_Analysis\\'
figdirec = datapath+'Img_Resnet_2D_result_figure\\'

savepath = './output/'
# tboardpath=savepath+'log/'

# writer = SummaryWriter(tboardpath+'IPM109Steady_SE_Resnet_experiment_221016_v1')
# writer = SummaryWriter(tboardpath+'IPM109Timevaryinghalf_SE_Resnet_experiment_221014_v1')
# tboardpath+'IPM109Steady_SE_Resnet_experiment_221011_v1'
# 텐서보드는 cmd창에서 
# > **tensorboard --logdir=\\\\147.47.239.149\\SHRM-Motor\\전장품\\모터 테스트베드_데이터 정리\\log** 
# 
# 치면됨( https://localhost:6006 )
# class_total = ['NOR','BIF1','BIF3','BOF','DE02','DG06','SE04','SE08','SIS100','SIS500']
class_total = ['BIF3', 'ECC080', 'NOR1', 'BOF3', 'DE02', 'DG06', 'SIS050']
substate = class_total

numstate = len(substate)
# ## 1. 데이터 로드
# path=datapath+"Img_IPM109steady\\"
#path = datapath+"Img_IPM109timevarying_half\\"
path = datapath+"Img_IPM109timevarying_HM_N2560_1280\\"
#
info_col_name=['state','cond','speed','tq','n']
# data_info = pd.DataFrame(columns=info_col_name)
Imglist = os.listdir(path)

for cc in range(len(class_total)):
    label_1 = np.zeros((1,), dtype=int)
    data_1 = np.zeros((224, 224, 3, 1), dtype=float)
    label_2 = np.zeros((1,), dtype=int)
    data_2 = np.zeros((224, 224, 3, 1), dtype=float)
    for ii in range(len(Imglist)):
        Imgname = Imglist[ii]
        # 2거나 1또는3
        Imgname_split = Imgname.split('_')

        if Imgname.find('.png') is not -1 and Imgname_split[2][5] == '0':
           #print(Imgname_split[-1].split('.')[0][0])
           last_num = int(Imgname_split[-1].split('.')[0][0])
           if Imgname.find(class_total[cc]) is not -1: # 해당하는 고장모드만 데이터 쌓는다
                phase = Imgname_split[2][5]
                cond = Imgname_split[4]
                speed = Imgname_split[5]
                loadtq = Imgname_split[6]
                print(phase+' '+cond+' '+speed+' '+loadtq)
                class_num = class_total.index(cond)
                n_ = Imgname_split[-1].split('.')[0]
                n_num = n_[n_.find('ni')+2:]
                # data_info_temp = pd.Series([class_num,cond,speed,loadtq,n_num])
                # data_info = pd.concat([data_info, data_info_temp.T], ignore_index = True, axis=0)
                img_array = np.fromfile(path+Imglist[ii], np.uint8)
                Img = cv2.imdecode(img_array, 0)

            # phase 2,3 찾기
                Imgname_split2 = Imgname.split(' ')[1]
                for j in range(len(Imglist)):
                    if Imglist[j].find(Imgname_split2) is not -1: # 포함
                        img_array = np.fromfile(path + Imglist[j], np.uint8)
                        phase = Imglist[j].split(' ')[0]
                        if phase[-1] == '1':
                            Img2 = cv2.imdecode(img_array, 0)
                        elif phase[-1] == '2':
                            Img3 = cv2.imdecode(img_array, 0)
                Img_resize = cv2.resize(Img,Imgshape)
                Img2_resize = cv2.resize(Img2, Imgshape)
                Img3_resize = cv2.resize(Img3, Imgshape)
                Img_comb = np.dstack((Img_resize,Img2_resize))
                Img_comb = np.concatenate((Img_comb, Img3_resize[:,:,np.newaxis]),axis=2)
                #Img_comb = np.dstack((Img,Img2))
                #Img_comb = np.concatenate((Img_comb, Img3[:,:,np.newaxis]),axis=2)
                if last_num == 1:
                    data_1 = np.concatenate((data_1,Img_comb[:,:,:,np.newaxis]),axis=3)
                    label_1 = np.append(label_1,class_num)
                elif last_num == 2 or last_num ==3:
                    data_2 = np.concatenate((data_2,Img_comb[:,:,:,np.newaxis]),axis=3)
                    label_2 = np.append(label_2,class_num)
        # data랑 label save
    label_1=label_1[1:];label_2=label_2[1:]
    data_1=np.transpose(data_1[:,:,:,1:],(3,2,0,1)); data_2=np.transpose(data_2[:,:,:,1:],(3,2,0,1))
    np.save(savepath+'data1_Timevarying100_'+class_total[cc]+str(class_num)+'.npy',data_1)
    np.save(savepath+'label1_Timevarying100_'+class_total[cc]+str(class_num)+'.npy',label_1)
    np.save(savepath+'data2_Timevarying100_'+class_total[cc]+str(class_num)+'.npy',data_2)
    np.save(savepath+'label2_Timevarying100_'+class_total[cc]+str(class_num)+'.npy',label_2)
