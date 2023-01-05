#!/usr/bin/env python
# coding: utf-8

import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
datadir = '\\\\147.47.239.143\\SHRM-Personal\\Personal_Drive\\박찬희\\RESEARCH_2022\\1. PYCODE\\Motor_Signal_Analysis\\final\\'
#datadir = '\\\\147.47.239.143\\SHRM-Personal\\Personal_Drive\\박찬희\\RESEARCH_2022\\1. PYCODE\\Motor_Signal_Analysis\\형민선별4final\\'
folder_list = os.listdir(datadir)
print(folder_list)
datapath = '\\\\147.47.239.143\\SHRM-Personal\\Personal_Drive\\박찬희\\RESEARCH_2022\\1. PYCODE\\Motor_Signal_Analysis\\'
Figsavedir = datapath+'figure_IPM109timevarying_HM_N2560_1280\\'
Figsavedir2 = datapath+'Img_IPM109timevarying_HM_N2560_1280\\'
datasavedir = datapath+'data\\'
'''Parameter setting
'''
muT=64 #현재는 임의의 값이지만나중에 muT계산해서 바꾸고 성능체크

def Load_file(datadir,folder,file_num):
    df=pd.read_csv(datadir+folder+'\\'+file_num,delimiter='\t',
    skiprows=range(49),header=None,encoding='cp949')
    log_df = df.loc[:,[0,1,5,9,25,29]]
    log_df.columns = ["Time","Ia", "Ib","Ic", "Va","SP"]
    return log_df, file_num

def DataLoad(path,folder,file_num):#file):
    # f =pd.read_csv(path+folder+'\\'+file, delimiter='\t', skiprows=48,dtype=None)
    f = pd.read_csv(path + folder + '\\' + file_num, delimiter='\t',
                     skiprows=range(48),dtype=None, encoding='cp949')
    Data=pd.DataFrame(np.zeros(len(f)-2))
    Data['Time']=f.iloc[:-2,0].values
    Data['Ia']=f.iloc[:-2,1]
    Data['Ib']=f.iloc[:-2,5]
    Data['Ic']=f.iloc[:-2,9]
    Data['TQ']=f.iloc[:-2,13]
    Data['Va']=f.iloc[:-2,17]
    Data['Vb']=f.iloc[:-2,21]
    Data['Vc']=f.iloc[:-2,25]
    if file_num=='220214_IPM_DE02_4500RPM_3.13N_Trapazoid(8,2,8)_param(50_0.1)_2.txt':
        Data['SP']=f.iloc[:-2,25]
    else:
        Data['SP']=f.iloc[:-2,29]
    del Data[0]
    return Data

def Load_Un_HM(f, file=''):
    # 속도 500이상구간만 추출
    startindex = [];
    endindex = [];
    if file == '220214_IPM_DE02_4500RPM_3.13N_Trapazoid(8,2,8)_param(50_0.1)_2.txt':
        Un = np.array(f['Ia'].iloc[50000:200000])
        Vn = np.array(f['Ib'].iloc[50000:200000])
        Wn = np.array(f['Ic'].iloc[50000:200000])

    else:
        for k in range(len(f) - 1):
            if (f['SP'].iloc[k] > 100) and (f['SP'].iloc[k + 1] < 100):
                endindex = endindex + [k];
            if (f['SP'].iloc[k] < 100) and (f['SP'].iloc[k + 1] > 100):
                startindex = startindex + [k + 1];
        if (len(endindex) == 1) and (len(startindex) == 2):  ## 잘못잘린 경우 보정
            startindex = startindex[0]; endindex =endindex[0];
            # Un = np.array(f['Ia'].iloc[startindex[0]:endindex[0]])
            # Vn=np.array(f['V_c'].iloc[startindex[0]:endindex[0]]).reshape(-1,1)
            # Wn=np.array(f['W_c'].iloc[startindex[0]:endindex[0]]).reshape(-1,1)
        elif (len(endindex) == 1) and (len(startindex) == 1):
            startindex = startindex[0]; endindex =endindex[0];
            # Un = np.array(f['U_c'].iloc[startindex[0]:endindex[0]])
            # Vn=np.array(f['V_c'].iloc[startindex[0]:endindex[1]]).reshape(-1,1)
            # Wn=np.array(f['W_c'].iloc[startindex[0]:endindex[1]]).reshape(-1,1)
        else:
            startindex = startindex[0]; endindex =endindex[1];
        #     Un = np.array(f['U_c'].iloc[startindex[0]:endindex[1]])
            # Vn=np.array(f['V_c'].iloc[startindex[0]:endindex[1]]).reshape(-1,1)
            # Wn=np.array(f['W_c'].iloc[startindex[0]:endindex[1]]).reshape(-1,1)
        log_df = f.iloc[startindex:endindex,:]
        log_df.reset_index(drop=True, inplace=True)
    return log_df, startindex,endindex# Un

def func_subplots_by_time_varying(col, df_total, Tick_time, savedir, figname, savemode):
    fig1, ax = plt.subplots(len(col)-1,1,figsize=(20,10))
    spfi=int(figname.split('_')[3])
    col_name = df_total.columns
    for c in range(len(col)-1):
        ax[c].plot(df_total["Time"],df_total.iloc[:,c+1])
        for t_tick_start in Tick_time:
            if col_name[c+1] == "SP":
                ax[c].axvline(x=df_total.iloc[t_tick_start, 0], ymin=0,ymax=spfi, color='b', linestyle='--')
            elif col_name[c + 1] == "TQ":  # speed만 왜 표기가 안되는지..min, max에서 이상한 값이 찍혔나보다
                ax[c].axvline(x=df_total.iloc[t_tick_start,0],ymin=0, ymax=4.7, color='b',linestyle='--')
            else:
                ax[c].axvline(x=df_total.iloc[t_tick_start,0],ymin=min(df_total.iloc[:,c+1]), ymax=max(df_total.iloc[:,c+1]), color='b',linestyle='--')
        ax[c].set_xlabel('Time'); ax[c].set_ylabel(col[c+1],rotation=0)
        ax[c].grid(True)
#     plt.show()
    if savemode==True:
        fig1.savefig(savedir+figname+'.jpg')
        plt.close(fig1)
    return fig1, ax

def func_subplots_constant(col, df_total, savemode, savedir, figname):
    fig, ax = plt.subplots(len(col)-1,1,figsize=(20,10))
    for c in range(len(col)-1):
        ax[c].plot(df_total["Time"],df_total.iloc[:,c+1])
        ax[c].set_xlabel('Time'); ax[c].set_ylabel(col[c+1],rotation=0);
        ax[c].grid(True)
     # plt.show()
    if savemode==True:
        fig.savefig(savedir+figname+'.jpg')
        plt.close(fig)
    return fig, ax

# phase fit
import scipy.signal
from matplotlib.pyplot import scatter
from scipy.signal import hilbert
from scipy.interpolate import interp1d
def phase_change(signal):
    """
    힐버트 방식을 통해 각 찾기/ 너무 길면 hilbert nan+nanj로 나옴
    """
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    phase_scaled = instantaneous_phase % (2 * np.pi) / (np.pi) - 1
    zero = []
    for k in range(len(phase_scaled) - 1):
        if (phase_scaled[k + 1] < -0.9) and (phase_scaled[k] > 0.9):
            if zero == []:
                zero.append(k)
            elif (k - zero[-1]) > 5:
                zero.append(k)
    return zero
def ResampleOT(Un, num, Show):
    """

    힐버트 방식을 통해 구한 각으로 데이터 리샘플링 진행(Cubic Polynominal)

    """
    total_signal = []
    zero_crossings = phase_change(Un)
    for k in range(len(zero_crossings) - 1):
        x = list(range(zero_crossings[k], zero_crossings[k + 1] + 1))
        y = list(Un[x])
        fq = interp1d(x, y, kind='cubic')
        xint = np.linspace(x[0], x[-1], num + 1)
        resampled_signal = list(fq(xint)[:-1])
        if k == 0:
            total_signal = resampled_signal
        else:
            total_signal.extend(resampled_signal)

    if Show == True:
        plt.figure(figsize=(12, 5))
        plt.plot(total_signal)
        plt.title('resampled signal muT: ' + str(num))
        plt.figure(figsize=(12, 5))
        plt.plot(Un)
        plt.title('original signal')
    return total_signal
# analytic 평면에 그림
def plot_analytic_signal(analytic_signal, figname, savedir, savemode=False,axisshow=True, Show=False):
    fig2=plt.figure(figsize=(4,4))
    plt.scatter(np.real(analytic_signal), np.imag(analytic_signal),marker='.')   # numpy에서 real과 imag 메소드 사용
    if axisshow:
        plt.ylabel('hilbert(X)')
        plt.xlabel('Real(X)')
    #     plt.axis([-20,20,-20,20])
        plt.axhline(y=0,color='black')
        plt.axvline(x=0, color='black')
        plt.grid(True)
        plt.title(figname)
    else:
        plt.axis('off')
    if savemode==True:
        fig2.savefig(savedir+figname+'.jpg')
        plt.close(fig2)
    if Show == True:
        plt.show()
        return fig2
def Img_analytic_plane(analytic_signal, figshape,figname, savedir, savemode=False,axisshow=False, Show=False):
    fig3=plt.figure(figsize=figshape)
    plt.scatter(np.real(analytic_signal), np.imag(analytic_signal),marker='.')   # numpy에서 real과 imag 메소드 사용
    if axisshow:
        plt.ylabel('hilbert(X)')
        plt.xlabel('Real(X)')
        plt.axhline(y=0,color='black')
        plt.axvline(x=0, color='black')
        plt.grid(True)
        plt.title(figname)
    else:
        plt.axis('off')
    if savemode==True:
        fig3.savefig(savedir+figname+'.png')
        plt.close(fig3)
    if Show == True:
        plt.show()
def AmpScaling(log_df, idx_samp):
    df_n = log_df.iloc[idx_samp, :]
    a_ = df_n.apply(lambda x: (x ** 2), axis=1)  # rms
    df_n_rms = a_.sum().pow(1 / 2)
    df_n_amp_scale = df_n.div(df_n_rms)
    return df_n_amp_scale, idx_samp
def revNSPmaking(RN,X_in,Y_in,ranE):
    temp_ICRPmtx_out = np.zeros((1,1,RN ,RN ),dtype=np.float);
    for n in range(RN):
        idx_n=np.where((X_in>ranE[n]) &(X_in<=ranE[n+1]));
        for mm in range(len(idx_n[0])):
            y_value=Y_in[idx_n[0][mm]];
            for k in range(RN):
                if (y_value > ranE[k]) and (y_value<=ranE[k+1]):
                    temp_ICRPmtx_out[0,0,n,k]+=1;
    # print(np.sum(np.sum(temp_ICRPmtx_out)))
    temp_ICRPmtx_out[0,0,:,:] = temp_ICRPmtx_out/np.sum(temp_ICRPmtx_out);
    # print(temp_ICRPmtx_out.shape)
    # print(a)
    # print(np.sum(temp_ICRPmtx_out))
    return temp_ICRPmtx_out
'''
# start for loop
'''
N=int(2560)
df_static_setname = ['mean','std','rms','shapefactor','skew','kurt']
figshape = (4,4)
cmaptime = 'magma'  # 'twilight_shifted'#'twilight'#'cividis'
cbar_num_format = '%.3f'
plt.rcParams.update({'font.size': 10})
plt.rcParams['font.family'] = 'Times New Roman'
for cond_name in folder_list: # 상태
    tanh_MTX_total = np.zeros((1, 3, RN, RN), dtype=float)
    label_total = np.zeros((1,),dtype=int)
    for file_name in os.listdir(datadir+cond_name):
        log_df_o = DataLoad(datadir, cond_name,file_name)
        figname1 = file_name[:-4]

        # index 조각별로 계산 # 변속에선 유효구간 먼저 잘라줌
        log_df, startindex, endindex=Load_Un_HM(log_df_o, file_name)
        log_df_len = len(log_df) # 전체 유효 길이
        # idx_samp_start = list(range(0,log_df_len-N,N))#no overlap np.arange(2000, 4000 - 1, 1)
        overlap_n = 1280
        idx_samp_start = list(range(0, log_df_len - N, overlap_n))  # no overlap np.arange(2000, 4000 - 1, 1)
        Nlen = len(idx_samp_start)
        for n_i in range(Nlen-1):
            # if len(np.setdiff1d(n_i,[int(Nlen//3),int(Nlen//2),int(Nlen//3*2)]))==0:# 다계산 안하고 몇개만
           print(str(n_i) + '/' + str(Nlen) + ' datafile: ' + figname1)
           # idx_samp = list(range(idx_samp_start[n_i], idx_samp_start[n_i + 1], 1))
           idx_samp = list(range(idx_samp_start[n_i], idx_samp_start[n_i]+N, 1))
                # 어느구간을 plot하는건지 원신호 plot
           t_fig_name = figname1 + 'ni' + str(n_i)+ '.png'
           temp_Tick_time = np.array([idx_samp[0], idx_samp[-1]])
           _, _ = func_subplots_by_time_varying(log_df.columns, log_df, temp_Tick_time, Figsavedir, t_fig_name,
                                                     savemode=True)
                # 1. scaling
           df_n_amp_scale,idx_samp = AmpScaling(log_df, idx_samp)
           RN = 50
                # 2. phase resamp
           tanh_MTX_ = np.zeros((1, 3, RN, RN), dtype=float)
           for pp in range(3):
                resampled_signal=ResampleOT(df_n_amp_scale.iloc[:,pp+1].to_numpy(),muT,Show=False) #
                plane_fig_name = '_analytic_plane' + str(pp) + ' ' + figname1 + 'ni' + str(n_i) + '.png'
                if not resampled_signal==[]:
                    # mtx에 넣기
                    tanhX = np.tanh(np.real(hilbert(resampled_signal)));
                    tanhY = np.tanh(np.imag(hilbert(resampled_signal)));
                    ranE = np.linspace(-1, 1, num=RN + 1, dtype=np.float)
                    # 기존 ICRM

                    tanh_MTX = revNSPmaking(RN, tanhX, tanhY, ranE)
                    tanh_MTX_[0,pp,:,:] = tanh_MTX
                    if pp == 0:
                        Vizfig = plt.figure(figsize=(12, 5))
                        Vizfig, axes = plt.subplots(1, 3)  # ,constrained_layout=True)
                    Vizfig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
                    ax = plt.gca()
                    axes[pp].imshow(tanh_MTX[0, 0, :, :], cmap=cmaptime)
                    im = axes[pp].imshow(tanh_MTX[0, 0, :, :], cmap=cmaptime)
                    divider = make_axes_locatable(axes[pp])
                    cax = divider.append_axes("right", size="2%", pad=0.1)
                    Vizfig.colorbar(im, cax=cax, ax=axes[pp])  # ,format = '%.6f')

                    Vizfig.subplots_adjust(top=0.8)
                    Vizfig.suptitle(figname1 + '\n' + ' tanhMTX scale ' + 'col abc phase', x=0.5, y=0.98)

                    if pp == 2:
                        plt.show()
                        Vizfig.savefig(Figsavedir + 'tanhMTX scale ' + plane_fig_name + '.png')

                    tanh_MTX_total = np.concatenate((tanh_MTX_total, tanh_MTX_), axis=0) #data = np.zeros((1,3,Imgshape[0], Imgshape[1]), dtype=float)
                    class_num = class_total.index(cond)
                    #plot_analytic_signal(hilbert(log_df.iloc[idx_samp,pp+1]),'fig_original_n'+plane_fig_name,Figsavedir, savemode = True,axisshow=True, Show=False )
                    #plot_analytic_signal(hilbert(resampled_signal), 'fig_scale+resamp'+plane_fig_name,Figsavedir, savemode = True,axisshow=True, Show=False)
                    #Img_analytic_plane(hilbert(resampled_signal), figshape,'Img' + plane_fig_name, Figsavedir2,savemode=True, axisshow=False, Show=False)
                else:
                    print(plane_fig_name + ' resamp is []')
''' #변속
# for loop
for fo in folder_list:
    file_list = os.listdir(datadir+fo)
    
    for tq_lev in range(len(file_list)):
        figname1=file_list[tq_lev]
        print(fo+' filenum: '+str(tq_lev)+'/'+str(len(file_list)))
        log_df, filename = Load_file(datadir,fo,tq_lev)
        speed=log_df.iloc[:,5]
        acc = np.diff(log_df.iloc[:,5])
        diff_acc = np.diff(acc)
    
        temp = np.where(abs(diff_acc)>0.001)
        Tick_time = np.array([temp[0][0], temp[0][-1]])
        print(log_df.iloc[Tick_time,0])
        fig1,ax1 = func_subplots_by_time(log_df.columns, log_df, Tick_time, Figsave, Figsavedir, figname1)
        fig = make_subplots(rows=5, cols=1,shared_xaxes=True)
        #                     vertical_spacing=0.02)
        seg_idx = range(Tick_time[0],Tick_time[-1])
        seg_time = log_df.iloc[Tick_time[0]:Tick_time[-1],0]
        fig.add_trace(go.Scatter(x=seg_time,y=log_df.iloc[seg_idx,1],mode='lines'),row=1, col=1)
        fig.add_trace(go.Scatter(x=seg_time,y=log_df.iloc[seg_idx,2],mode='lines'),row=2, col=1)
        fig.add_trace(go.Scatter(x=seg_time,y=log_df.iloc[seg_idx,3],mode='lines'),row=3, col=1)
        fig.add_trace(go.Scatter(x=seg_time,y=log_df.iloc[Tick_time[0]:Tick_time[-1],4],mode='lines'),row=4, col=1)
        fig.add_trace(go.Scatter(x=seg_time,y=log_df.iloc[Tick_time[0]:Tick_time[-1],5],mode='lines'),row=5, col=1)

        fig.update_layout(height=1200, width=800, title = figname1, legend =None)
#         fig.show()
        fig.write_html(Figsavedir+figname1+".html")        
'''