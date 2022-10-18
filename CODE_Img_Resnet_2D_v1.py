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
class_total = ['NOR','BIF1','BIF3','BOF','DE02','DG06','SE04','SE08','SIS100','SIS500']
# class_total = ['BIF3', 'ECC080', 'NOR1', 'BOF3', 'DE02', 'DG06', 'SIS050']
substate = class_total

numstate = len(substate)
# ## 1. 데이터 로드
path=datapath+"Img_IPM109steady\\"
# path = datapath+"Img_IPM109timevarying_half\\"
# path = datapath+"Img_IPM109timevaring\\"
#
label = np.zeros((1,),dtype=int)
data = np.zeros((32,32,3,1),dtype=float)
info_col_name=['state','cond','speed','tq','n']
# data_info = pd.DataFrame(columns=info_col_name)
Imglist = os.listdir(path)
ii=0
for ii in range(len(Imglist)):
    Imgname = Imglist[ii]
    Imgname_split = Imgname.split('_')
    if Imgname.find('.png') is not -1 and Imgname_split[2][5]=='0':
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
        # Img_resize = cv2.resize(Img,Imgshape)
        # Img2_resize = cv2.resize(Img2, Imgshape)
        # Img3_resize = cv2.resize(Img3, Imgshape)
        # Img_comb = np.dstack((Img_resize,Img2_resize))
        # Img_comb = np.concatenate((Img_comb, Img3_resize[:,:,np.newaxis]),axis=2)
        Img_comb = np.dstack((Img,Img2))
        Img_comb = np.concatenate((Img_comb, Img3[:,:,np.newaxis]),axis=2)
        data = np.concatenate((data,Img_comb[:,:,:,np.newaxis]),axis=3)
        label = np.append(label,class_num)
# data랑 label save
label=label[1:]
data=np.transpose(data[:,:,:,1:],(3,2,0,1))
np.save(savepath+'data_Timevarying109steady_40x40.npy',data)
np.save(savepath+'label_Timevarying109steady_40x40.npy',label)
data = np.load(savepath+'data_Timevarying109steady.npy')
label = np.load(savepath+'label_Timevarying109steady.npy')
# resize
from skimage.transform import resize, rescale
shape_ = (Imgshape[0],Imgshape[1],3)
data_resize = np.zeros((data.shape[0],Imgshape[0],Imgshape[1],data.shape[1]),dtype=float)
for ii in range(data.shape[0]):
    x_t = np.transpose(data[ii,:],(1,2,0))
    data_resize[ii,:] = resize(x_t,shape_)
data = data_resize

# data balance 맞추고 나누기
def pick_num_from_x(x,num):
    siz = x.shape[0] # 전체 길이
    num_train = int(num)
    idx_perm = np.random.choice(siz,size=num,replace=False)
    return idx_perm

label_list, label_num = np.unique(label, return_counts = True)
num_per_label =min(label_num)
idx_test =[]; idx_train=[];
for label_ in label_list:
    label_idx = np.where(label==label_)
    temp_idx_train = label_idx[0][pick_num_from_x(label_idx[0],int(num_per_label*0.9))]
    temp_idx_test = np.setdiff1d(label_idx[0],temp_idx_train)
    idx_test.append(temp_idx_test)
    idx_train.append(temp_idx_train)

idx_train_flatten = [y for x in idx_train for y in x]
idx_test_flatten = [y for x in idx_test for y in x]
train_label=label[idx_train_flatten]
train_data=data[idx_train_flatten,:]

test_label=label[idx_test_flatten]
test_data=data[idx_test_flatten,:]

del data, label
# ### 데이터 셔플
def random_sampling(dataset, labels, num=None):
    permutation = np.random.permutation(labels.shape[0])
    if not num==None:
#         print("\n {} sampling".format(num))
        permutation = permutation[0:num]
#     else:
#         print("\nNo Down-Sampling")
    shuffled_dataset = dataset[permutation,:]#.reshape(permutation.shape[0],-1)
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

train_x_s,train_y_s,=random_sampling(train_data,train_label)
test_x_s,test_y_s,=random_sampling(test_data,test_label)

train_y_sn=torch.tensor(train_y_s)
test_y_sn=torch.tensor(test_y_s)

train_x_sn=torch.tensor(train_x_s)
test_x_sn=torch.tensor(test_x_s)

print('Training set', train_x_sn.shape,train_y_sn.shape)
print('Test set', test_x_sn.shape, test_y_sn.shape)

## 파이토치 입력 데이터로 변환
class MotorDataset():
    def __init__(self, data, label):#, transforms=None):
        self.dataset=data.float()
        self.label=label
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data_idx=self.dataset[idx,:]
        label_idx=self.label[idx]
        # if self.transform is not None:
        #     data_idx = self.transform(data_idx)
        return data_idx, label_idx

TRAIN_DATASET=MotorDataset(train_x_sn,train_y_sn)#,prep_transform)
TEST_DATASET=MotorDataset(test_x_sn,test_y_sn)#,prep_transform)

batch_size = 16

train_loader = DataLoader(dataset=TRAIN_DATASET, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(dataset=TEST_DATASET, batch_size=len(TEST_DATASET), shuffle=False)

from collections.abc import Iterable
print(issubclass(DataLoader, Iterable))
print('train loader length: ', len(train_loader)) # same as len(dataset) // batch_size
print('test loader length: ', len(test_loader))

# ## 2. Model
conditionname = 'Img_IPM109Steady'
# %% model
# Convolution - Batch Normalization - Activation - Dropout - Pooling 순서
class GlobalAvgPool2D(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2D, self).__init__()

    def forward(self, x):
        return x.mean(axis=-1).mean(axis=-1)


def xavier_uniform_weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear]:
        m.weight.data = torch.nn.init.xavier_normal_(m.weight.data)
        # m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data)
        # m.bias.data = torch.nn.init.xavier_uniform_(m.bias.data)


class IdentityPadding(nn.Module):
    def __init__(self, in_channels, filter_size, stride):
        super(IdentityPadding, self).__init__()

        self.pooling = nn.MaxPool2d(kernel_size=1, stride=stride)
        self.add_channels = filter_size - in_channels

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        # IdentityPadding에서는 zero padding을 수행합니다.
        # F.pad는 padding 값을 주는 함수이고 이는 feature map의 마지막 축에 대해(0, 0)으로 padding,
        # 마지막에서 두 번째 축에 대해서 (0, 0), 세 번째 축에 대해 (0, self.add_channel)만큼 padding을 하라는 의미입니다.
        out = self.pooling(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filter_size, stride=2):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, filter_size, kernel_size=5, stride=2, padding=(5 - 1) // 2, dilation=1)
        )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        # self.layer2 = nn.Sequential(
        #             nn.BatchNorm2d(filter_size),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(filter_size,filter_size,kernel_size=3,stride=1,padding=(3-1)//2,dilation=1)
        #             )
        # self.layer3 = nn.Sequential(
        #             nn.BatchNorm2d(filter_size),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(filter_size,filter_size,kernel_size=3,stride=1,padding=(3-1)//2,dilation=1)
        #             )
        if stride == 2:
            self.down_sample = IdentityPadding(in_channels, filter_size, stride)
        else:
            self.down_sample = None

    def forward(self, x):
        shortcut = x
        out = self.layer1(x)
        # out11 = self.layer2(out1)
        # out= self.layer3(out11)

        if self.down_sample is not None:
            shortcut = self.down_sample(x)
            # print(out1.shape)
        # print(out.shape)
        # print(shortcut.shape)
        out += shortcut
        out = self.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self, in_channels, filter_size, numstate, stride=2):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filter_size // 2, kernel_size=5, stride=2, padding=(5 - 1) // 2,
                               bias=True)

        self.res_block1 = ResidualBlock(filter_size // 2, filter_size, stride=2)
        self.res_block2 = ResidualBlock(filter_size, filter_size, stride=2)
        self.res_block3 = ResidualBlock(filter_size, filter_size, stride=2)
        self.gap = GlobalAvgPool2D()
        self.linear = nn.Linear(filter_size, numstate)

    def forward(self, x):
        out0 = self.conv1(x)
        # print(out0.shape)
        out1 = self.res_block1(out0)
        out2 = self.res_block2(out1)
        out3 = self.res_block3(out2)

        y_flat = self.gap(out3)
        # print(y_flat.shape)
        y = self.linear(y_flat)
        return y, out0, out1, out2, out3


in_channels = channel
filter_size = Imgshape[0]
numstate = len(np.unique(label))
cnn_model = Resnet(in_channels, filter_size, numstate).cuda()

def multi_acc(y_pred, y_test):
    y_pred_tags = []
    y_pred = y_pred.cpu()
    y_test = y_test.cpu()

    # crossentropy loss일때
    y_pred = y_pred.argmax(dim=1)
    y_pred_tags = torch.Tensor(y_pred.float())

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 1000) / 10

    return acc, y_pred_tags

def modeltrain(model, criterion, optimizer, epochs, train_loader=train_loader, valid_loader=test_loader):
    # 출력
    last = {"loss": sys.float_info.max}
    accuracy_stats = {'train': [], "val": []}
    loss_stats = {'train': [], "val": []}
    min_val_loss = np.Inf;
    fin_epoch = num_epochs
    for e in range(1, epochs+1):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch = X_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)

            optimizer.zero_grad()

            y_train_pred, *_ = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch.long())
            train_acc, y_train_tags = multi_acc(y_train_pred, y_train_batch.float())

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

            # VALIDATION
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in valid_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred, *_ = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch.long())
                val_acc, y_val_tags = multi_acc(y_val_pred, y_val_batch.float())
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        train_acc_ = train_epoch_acc / len(train_loader)
        val_acc_ = val_epoch_acc / len(valid_loader)
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(valid_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(valid_loader))
        if val_acc_ >= np.max(accuracy_stats['val']):
            # print(val_acc_)
            last["state"] = model.state_dict()
            last["train_loss"] = train_epoch_loss
            last["val_loss"] = val_epoch_loss
            last["val_acc"] = val_acc_
            last["train_acc"] = train_acc_
            last["epoch"] = e
            last["test_result"] = y_val_tags
            last["test_label"] = y_val_batch
            # if val_epoch_loss/len(valid_loader) < min_val_loss:
        # # if train_acc > min_train_acc:
        #       epochs_no_improve = 0
        #       min_val_loss = val_epoch_loss/len(valid_loader)
        #       torch.save(model.state_dict(), PATH+conditionname+'.pth')
        # else:
        #     epochs_no_improve += 1

        # ## early stopping by testing loss
        # if epochs_no_improve >= patience and train_epoch_acc/len(train_loader)>99.5:
        #     print('Early stop!')
        #     ##early_stop = True
        #     fin_epoch = e
        #     break
        # else:
        #     if e % 20 == 0:
        #         print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.3f} | Val Loss: {val_epoch_loss/len(valid_loader):.3f} | Train Acc: {train_epoch_acc/len(train_loader):.1f}| Val Acc: {val_epoch_acc/len(valid_loader):.1f}')
        #     fin_epoch = e
        #     continue
        # break
        if e % 50 ==0:
            print(
                f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.3f} | Val Loss: {val_epoch_loss / len(valid_loader):.3f} | Train Acc: {train_epoch_acc / len(train_loader):.1f}| Val Acc: {val_epoch_acc / len(valid_loader):.1f}')
        if e == 1:
            torch.save(model.state_dict(), savepath + 'model/' + conditionname+' model.pth')
            fin_epoch = e
        elif train_epoch_loss / len(train_loader) < loss_stats['train'][-1]:
            torch.save(model.state_dict(), savepath+ 'model/' + conditionname+' model.pth')
            fin_epoch = e
        elif e == num_epochs:
            torch.save(model.state_dict(), savepath + 'model/' + conditionname + ' model.pth')
            fin_epoch = e

    return last, loss_stats, accuracy_stats, fin_epoch


# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=l2norm)

last, loss_stats, accuracy_stats, fin_epoch = modeltrain(model=cnn_model, optimizer=optimizer, criterion=criterion, epochs=num_epochs)

with open(savepath + 'last_info_' + conditionname + '.pkl', 'wb') as f:
    pickle.dump(last,f, pickle.HIGHEST_PROTOCOL)
    # last = pickle.load(f)

print("Optimization Finished!")


def plot_acc_loss(step, trainset, testset, figdirec, condname, fontsize, savemode=False):
    ep = np.array(range(step))
    plt.rcParams.update({'font.size': fontsize})
    plt.plot(ep, trainset[:, 0], 'b')
    plt.plot(ep, testset[:, 0], 'r')
    trainmean = np.mean(trainset[-10:-1, :], axis=0)
    testmean = np.mean(testset[-10:-1, :], axis=0)

    plt.xlabel('Epoch ' + str(round(trainmean[0], 2)) + ' ' + str(round(testmean[0], 2)))
    plt.ylabel('Accuracy')
    plt.title("\n".join(wrap(condname, 60)))

    plt.grid(True)
    if savemode:
        plt.savefig(figdirec + "Acc " + condname + ".png")
    plt.show()

    plt.plot(ep, trainset[:, 1], 'b')
    plt.plot(ep, testset[:, 1], 'r')
    plt.xlabel('Epoch ' + str(round(trainmean[1], 2)) + ' ' + str(round(testmean[1], 2)))
    plt.ylabel('Loss')
    plt.title("\n".join(wrap(condname, 60)))
    plt.grid(True)
    if savemode == 1:
        plt.savefig(figdirec + 'Loss ' + condname + '.png')
    plt.show()
    return trainmean, testmean


def plot_confusion_matrix(class_, label_test, y_pred_test, trainmean, testmean, figdirec, conditionname,
                          normalize=False, savemode=False):
    class_names = np.asarray(class_)
    classes = class_names
    label_test_ = label_test
    cnf_matrix = confusion_matrix(label_test_, y_pred_test)  # label test , predict test
    np.set_printoptions(precision=2)
    cm = cnf_matrix
    cmap = plt.cm.Blues
    fig = plt.figure(figsize=(5, 5))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.title("\n".join(wrap(conditionname + " Train: " + str(round(trainmean, 2)) + ", Test: " + str(round(testmean, 2)), 60)))
        # plt.title("Train: "+str(round(trainmean,2))+", Test: "+str(round(testmean,2)));#Normalized confusion matrix")
    else:
        plt.title("\n".join(
            wrap(conditionname + " Train: " + str(round(trainmean, 2)) + ", Test: " + str(round(testmean, 2)), 60)))
        # plt.title("Train: "+str(round(trainmean,2))+", Test: "+str(round(testmean,2)));#Normalized confusion matrix")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylim([len(classes) - 0.5, -0.5])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.rcParams.update({'font.size': 20})
    plt.show()

    if savemode:
        if normalize:
            fig.savefig(figdirec + "CFmtx norm-" + conditionname + ".png")
        else:
            fig.savefig(figdirec + "CFmtx " + conditionname + ".png")


train_acc_loss_set = np.column_stack([np.array(accuracy_stats['train']), np.array(loss_stats['train'])])
test_acc_loss_set = np.column_stack([np.array(accuracy_stats['val']), np.array(loss_stats['val'])])

trainresult, testresult = plot_acc_loss(fin_epoch, train_acc_loss_set, test_acc_loss_set, figdirec, conditionname, 12,
                                        savemode=True)

# test acc max/	@ test loss/train acc max/ @ train loss /
# acc mu/	loss mu	// 	train acc mu

# AccLoss_best_avg[iter_iiii, iter_i, 0, cond_i] = last["val_acc"]
# AccLoss_best_avg[iter_iiii, iter_i, 1, cond_i] = last["val_loss"]
# AccLoss_best_avg[iter_iiii, iter_i, 2, cond_i] = last["train_acc"]
# AccLoss_best_avg[iter_iiii, iter_i, 3, cond_i] = last["train_loss"]
# AccLoss_best_avg[iter_iiii, iter_i, 4:6, cond_i] = testresult
# AccLoss_best_avg[iter_iiii, iter_i, 6:8, cond_i] = trainresult
# AccLoss_best_avg[iter_iiii, iter_i, 8, cond_i] = endtime
# %% CFmtx
# tmodel_label = te_output.argmax(dim=1)
# te_x= last["
# te_label = torch.from_numpy(Label_test).float();
# te_label =te_label.cuda()
# te_output, *_= cnn_model(te_x)
# test_acc, t_tag= multi_acc(te_output, te_label.long())
# test_tag = t_tag.cpu().detach().numpy();
t_tag = last["test_result"].cpu().detach().numpy()  # 결과
te_label_cpu = last["test_label"].cpu().detach().numpy()  # 라벨
plot_confusion_matrix(substate, te_label_cpu, t_tag, last["train_acc"], last["val_acc"], figdirec,
                      'Test ' + conditionname, normalize=False, savemode=True)
