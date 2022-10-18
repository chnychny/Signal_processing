# IAML 환경에서 구동 성동: timm(torch 1.12) 깔고 pip으로 pytorch 1.7+cu10.1 설치함
# pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import itertools
import numpy as np
from six.moves import cPickle as pickle
import sys
from sklearn.metrics import confusion_matrix
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
Imgshape = (32,32)
channel = 3
l2norm = 0.001;
num_epochs =200;
batch_size = 64;
learning_rate = 1e-4;

# from torch.utils.tensorboard import SummaryWriter
datapath = '\\\\147.47.239.143\\SHRM-Personal\\Personal_Drive\\박찬희\\RESEARCH_2022\\1. PYCODE\\Motor_Signal_Analysis\\'
figdirec = datapath+'Img_Resnet_2D_result_figure\\'
savepath = './output/'

class_total = ['NOR','BIF1','BIF3','BOF','DE02','DG06','SE04','SE08','SIS100','SIS500']
# class_total = ['BIF3', 'ECC080', 'NOR1', 'BOF3', 'DE02', 'DG06', 'SIS050']
substate = class_total

numstate = len(substate)
# ## 1. 데이터 로드
path=datapath+"Img_IPM109steady\\"
# path = datapath+"Img_IPM109timevarying_half\\"
# path = datapath+"Img_IPM109timevaring\\"

data = np.load(savepath+'data_Timevarying109steady.npy')
label = np.load(savepath+'label_Timevarying109steady.npy')
# resize
# from skimage.transform import resize, rescale
# shape_ = (Imgshape[0],Imgshape[1],3)
# data_resize = np.zeros((data.shape[0],Imgshape[0],Imgshape[1],data.shape[1]),dtype=float)
# for ii in range(data.shape[0]):
#     x_t = np.transpose(data[ii,:],(1,2,0))
#     data_resize[ii,:] = resize(x_t,shape_)
# data = data_resize

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
del train_data, train_label, test_data, test_label

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

train_loader = DataLoader(dataset=TRAIN_DATASET, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=TEST_DATASET, batch_size=len(TEST_DATASET), shuffle=False)

from collections.abc import Iterable
print(issubclass(DataLoader, Iterable))
print('train loader length: ', len(train_loader)) # same as len(dataset) // batch_size
print('test loader length: ', len(test_loader))

# ## 2. Model
numstate = len(substate)

# %% model
import timm
avail_pretrained_models = timm.list_models(pretrained=True)
model_list = ['vit_base_patch16_224','ecaresnet101d','inception_resnet_v2','seresnet152d']
m=0;
header = 'Img_IPM109Steady_32x32_'
learn_condition = 'b'+str(batch_size)+'_ep'+str(num_epochs)+'adaml2norm'+str(l2norm)
conditionname = header+model_list[m]+'_'+learn_condition

model = timm.create_model(model_list[m], img_size=Imgshape[0], pretrained=True,num_classes=numstate)
in_channels = channel
cnn_model = model.cuda()

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
    accuracy_stats = {'acctrain': [], "accval": [],'losstrain': [], "lossval": []}
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

            y_train_pred = model(X_train_batch)
            # probabilities = torch.nn.functional.softmax(y_train_pred, dim=0)
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
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch.long())
                val_acc, y_val_tags = multi_acc(y_val_pred, y_val_batch.float())

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        train_acc_ = train_epoch_acc / len(train_loader)
        val_acc_ = val_epoch_acc / len(valid_loader)
        accuracy_stats['losstrain'].append(train_epoch_loss / len(train_loader))
        accuracy_stats['lossval'].append(val_epoch_loss / len(valid_loader))
        accuracy_stats['acctrain'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['accval'].append(val_epoch_acc / len(valid_loader))
        if val_acc_ >= np.max(accuracy_stats['accval']):
            # print(val_acc_)
            last["state"] = model.state_dict()
            last["train_loss"] = train_epoch_loss
            last["val_loss"] = val_epoch_loss
            last["val_acc"] = val_acc_
            last["train_acc"] = train_acc_
            last["epoch"] = e
            last["test_result"] = y_val_tags
            last["test_label"] = y_val_batch

        # break
        if e % 50 ==0:
            print(
                f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.3f} | Val Loss: {val_epoch_loss / len(valid_loader):.3f} | Train Acc: {train_epoch_acc / len(train_loader):.1f}| Val Acc: {val_epoch_acc / len(valid_loader):.1f}')
        if e == 1:
            torch.save(model.state_dict(), savepath + 'model/' + conditionname+' model.pth')
            fin_epoch = e
        elif train_epoch_loss / len(train_loader) < accuracy_stats['losstrain'][-1]:
            torch.save(model.state_dict(), savepath+ 'model/' + conditionname+' model.pth')
            fin_epoch = e
        elif e == num_epochs:
            torch.save(model.state_dict(), savepath + 'model/' + conditionname + ' model.pth')
            fin_epoch = e

    return last,  accuracy_stats, fin_epoch

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=l2norm)

# 학습수행부분
# criterion = nn.BCEWithLogitsLoss()
last, accuracy_stats, fin_epoch = modeltrain(model=cnn_model, optimizer=optimizer, criterion=criterion, epochs=num_epochs)

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


train_acc_loss_set = np.column_stack([np.array(accuracy_stats['acctrain']), np.array(accuracy_stats['losstrain'])])
test_acc_loss_set = np.column_stack([np.array(accuracy_stats['accval']), np.array(accuracy_stats['lossval'])])

trainresult, testresult = plot_acc_loss(fin_epoch, train_acc_loss_set, test_acc_loss_set, figdirec, conditionname, 12,
                                        savemode=True)
t_tag = last["test_result"].cpu().detach().numpy()  # 결과
te_label_cpu = last["test_label"].cpu().detach().numpy()  # 라벨
plot_confusion_matrix(substate, te_label_cpu, t_tag, last["train_acc"], last["val_acc"], figdirec,
                      'Test ' + conditionname, normalize=False, savemode=True)
