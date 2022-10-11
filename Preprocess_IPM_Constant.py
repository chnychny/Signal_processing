#!/usr/bin/env python
# coding: utf-8

import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
datadir = '\\\\147.47.239.143\\SHRM-Personal\\Personal_Drive\\박찬희\\RESEARCH_2022\\1. PYCODE\\Motor_Signal_Analysis\\\IPM109steady_mpan_final\\'#final\\'
folder_list = os.listdir(datadir)
print(folder_list)
datapath = '\\\\147.47.239.143\\SHRM-Personal\\Personal_Drive\\박찬희\\RESEARCH_2022\\1. PYCODE\\Motor_Signal_Analysis\\'
Figsavedir = datapath+'figure_IPM109steady\\'
Figsavedir2 = datapath+'Img_IPM109steady\\'
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
def func_subplots_by_time_varying(col, df_total, Tick_time, savedir, figname, savemode):
    fig1, ax = plt.subplots(len(col)-1,1,figsize=(20,10))
    for c in range(len(col)-1):
        ax[c].plot(df_total["Time"],df_total.iloc[:,c+1])
        for t_tick_start in Tick_time:
            ax[c].axvline(x=log_df.iloc[t_tick_start,0],ymin=min(df_total.iloc[:,c+1]), ymax=max(df_total.iloc[:,c+1]), color='r',linestyle='--')
        ax[c].set_xlabel('Time'); ax[c].set_ylabel(col[c+1],rotation=0)
        ax[c].grid(True)
#     plt.show()
    if savemode==True:
        fig1.savefig(savedir+figname+'.png')
        plt.close(fig1)
    return fig1, ax
def func_subplots_constant(col, df_total, savemode, savedir, figname):
    fig, ax = plt.subplots(len(col)-1,1,figsize=(20,10))
    for c in range(len(col)-1):
        ax[c].plot(df_total["Time"],df_total.iloc[:,c+1])
        ax[c].set_xlabel('Time'); ax[c].set_ylabel(col[c+1],rotation=0);
        ax[c].grid(True)
#     plt.show()
    if savemode==True:
        fig.savefig(savedir+figname+'.png')
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
        fig2.savefig(savedir+figname+'.png')
        plt.close(fig2)
    if Show == True:
        plt.show()
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
'''
# start for loop
'''
N=int(1280*2)
df_static_setname = ['mean','std','rms','shapefactor','skew','kurt']
figshape = (4,4)
for cond_name in folder_list: # 상태
    for file_name in os.listdir(datadir+cond_name):
        log_df, _ = Load_file(datadir, cond_name,file_name)
        figname1 = file_name[:-4]
        # total static
        df_n_static_set = log_df.describe()
        a_ = log_df.apply(lambda x: (x**2), axis=1) # rms
        abs_ = log_df.abs()
        # df_total_static_set = pd.DataFrame([log_df.mean(), log_df.std(), a_.sum().pow(1/2), a_.sum().pow(1/2).div(log_df.abs().mean()),
        #                                     log_df.skew(), log_df.kurt()]).T
        # df_total_static_set.columns = df_static_setname
        # df_total_static_set.to_csv(datasavedir+'pd_data\\df_total_static_set_'+figname1+'.csv')
        # df_total_static_set.to_pickle(datasavedir+'pd_data\\df_total_static_set_'+figname1+'.pkl')
        # index 조각별로 계산
        log_df_len = len(log_df) # 전체 유효 길이
        idx_samp_start = list(range(0,log_df_len-N,N))#no overlap np.arange(2000, 4000 - 1, 1)
        Nlen = len(idx_samp_start)
        for n_i in range(Nlen-1):
            # if len(np.setdiff1d(n_i,[int(Nlen//3),int(Nlen//2),int(Nlen//3*2)]))==0:# 다계산 안하고 몇개만
                # 1. scaling
            print(str(n_i)+'/'+str(Nlen)+' datafile: '+figname1)
            idx_samp = list(range(idx_samp_start[n_i],idx_samp_start[n_i+1],1))
            # 어느구간을 plot하는건지 원신호 plot
            t_fig_name = figname1 + 'ni' + str(n_i)+ '.png'
            temp_Tick_time = np.array([idx_samp[0], idx_samp[-1]])
            _, _ = func_subplots_by_time_varying(log_df.columns, log_df, temp_Tick_time, Figsavedir, t_fig_name,
                                                 savemode=True)

            df_n_amp_scale,idx_samp = AmpScaling(log_df, idx_samp)

            # 2. phase resamp
            for pp in range(3):
                resampled_signal=ResampleOT(df_n_amp_scale.iloc[:,pp+1].to_numpy(),muT,Show=False) #

                plane_fig_name ='_analytic_plane' +str(pp)+' '+figname1+'ni'+str(n_i)+'.png'
                # plot_analytic_signal(hilbert(log_df.iloc[idx_samp,pp+1]),'fig_original_n'+plane_fig_name,Figsavedir, savemode = True,axisshow=True, Show=False )
                # plot_analytic_signal(hilbert(resampled_signal), 'fig_scale+resamp'+plane_fig_name,Figsavedir, savemode = True,axisshow=True, Show=False)
                Img_analytic_plane(hilbert(resampled_signal), figshape,'Img' + plane_fig_name, Figsavedir2,savemode=True, axisshow=False, Show=False)
