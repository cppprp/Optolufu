import os

import numpy as np

import tool_functions as tf
import pandas as pd
f1_0 = '/Volumes/Optolufu_2/Optolufu_II_Fibrosis/fibrosis_run_I+CNs/2023_03_21_baseline/optolufu/analysis/'
f2_0 = '/Volumes/Optolufu_2/Optolufu_II_Fibrosis/fibrosis_run_II/2023_04_05_baseline/Optolufu/analysis/'
f1_7 = '/Volumes/Optolufu_2/Optolufu_II_Fibrosis/fibrosis_run_I+CNs/2023_03_29_day7/analysis/scaled/'
f2_7 = '/Volumes/Optolufu_2/Optolufu_II_Fibrosis/fibrosis_run_II/2023_04_12_day7/analysis/scaled/'
#f1_14
#f2_14
ast_0 = '/Volumes/Optolufu_2/Optolufu_asthma/asthma_run_II/2023_04_11_baseline/Optolufu/analysis/'
ast_7 = '/Volumes/Optolufu_2/Optolufu_asthma/asthma_run_II/2023_04_18_7days/analysis/'
ast_14 = '/Volumes/Optolufu_2/Optolufu_asthma/asthma_run_II/2023_04_25_14days/Optolufu/analysis/'


f1_0 = tf.getListOfFiles(f1_0, '', False, True)
f2_0 = tf.getListOfFiles(f2_0, '', False, True)
ast_0 = tf.getListOfFiles(ast_0, '', False, True)
ast_7 = tf.getListOfFiles(ast_7, '', False, True)
ast_14 = tf.getListOfFiles(ast_14, '', False, True)
f1_7=tf.getListOfFiles(f1_7, '', False, True)
f2_7=tf.getListOfFiles(f2_7, '', False, True)
#f1_14=tf.getListOfFiles(fibr_1_7, '', False, True)
#f2_14=tf.getListOfFiles(fibr_2_7, '', False, True)
list = [f1_0, f2_0, f1_7, f2_7, ast_0, ast_7, ast_14]
list = np.concatenate(list).ravel()
M = []
Mouse = None
for mouse in list:
    if Mouse is None:
        Mouse = [mouse]
    else:
        Mouse.append(mouse)
    files = tf.getListOfFiles(mouse+'/', '.csv', False, False)
    X = None
    Label = None

    for file in files[0:1]:

        print(file)
        df = pd.read_csv(file)
        buf = np.asarray(df['Ti [ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        if X is None:
            X = buf
            Label = ['Ti [ms]']

        else:
            X = np.concatenate((X,buf), axis= 0)
            Label.append('Ti [ms]')


        buf = np.asarray(df['Ti_slope']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X, buf), axis=0)
        Label.append('Ti_slope')

        buf = np.asarray(df['Ti_AuC [mm*ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X,buf), axis= 0)
        Label.append('Ti_AuC [mm*ms]')

        buf = np.asarray(df['Tp_[ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X,buf), axis= 0)
        Label.append('Tp_[ms]')

        buf = np.asarray(df['Tp_AuC [mm*ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X,buf), axis= 0)
        Label.append('Tp_AuC [mm*ms]')

        buf = np.asarray(df['Te [ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X, buf), axis=0)
        Label.append('Te [ms]')

        buf = np.asarray(df['Te_AuC [mm*ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X, buf), axis=0)
        Label.append('Te_AuC [mm*ms]')

        buf = np.asarray(df['Te_FE [ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X, buf), axis=0)
        Label.append('Te_FE [ms]')

        buf = np.asarray(df['Te_FE_slope']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X, buf), axis=0)
        Label.append('Te_FE_slope')

        buf = np.asarray(df['Te_FE_AuC [mm*ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X, buf), axis=0)
        Label.append('Te_FE_AuC [mm*ms]')

        buf = np.asarray(df['Te_SE [ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X, buf), axis=0)
        Label.append('Te_SE [ms]')

        buf = np.asarray(df['Te_SE_T1/2 [ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X, buf), axis=0)
        Label.append('Te_SE_T1/2 [ms]')

        buf = np.asarray(df['Te_SE_AuC [mm*ms]']).flatten()
        buf = buf[np.where(np.isfinite(buf))]
        buf = np.average(buf)
        buf = np.expand_dims(buf, axis=0)
        X = np.concatenate((X, buf), axis=0)
        Label.append('Te_SE_AuC [mm*ms]')


        #X = [item for sublist in X for item in sublist]
        #print(X)
        M.append(X)
df = pd.DataFrame(M)
df.index = Mouse
df.to_csv('/Users/angelika.svetlove/research files/Optolufu_fibrosis/test/cor_out_av_progress.csv', header=Label, index=True)




