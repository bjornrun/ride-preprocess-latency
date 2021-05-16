import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def _prepare_predict_latency(file_modem, file_bw, save_path):
    df = pd.read_csv(file_modem, index_col='datetime', parse_dates=True, low_memory=False).fillna(0)

    df_latence = pd.read_csv(file_bw, index_col='datetime', parse_dates=True).fillna(0)

    df_latence = df_latence[df_latence.index < dt.datetime(2020, 11, 1)]

    cells = df[(df['cellid'] != 0) & (df['cellid'] != '0') & (df['cellid'] != 'FFFFFFFF')][['cellid']]

    start_index = cells.index[0]
    current_cellid = cells.iloc[0].cellid
    previous_cellid = ''
    cells2 = pd.DataFrame([(cells.iloc[0].cellid, cells.iloc[0].cellid, 0, df_latence['5G_modem_latency'].mean())],
                          columns=['cellid', 'previous_cellid', 'dt', 'latence'], index=[start_index])

    timetransition2 = dict()
    numtransition2 = dict()
    lattransition2 = dict()
    timetransition3 = dict()
    numtransition3 = dict()
    lattransition3 = dict()

    for index, row in cells.iterrows():
        if row['cellid'] != current_cellid:
            tmp_latence = df_latence[(df_latence.index >= start_index) & (df_latence.index < index)][
                ['5G_modem_latency']].mean()
            if pd.isna(tmp_latence['5G_modem_latency']):
                tmp_latence = df_latence['5G_modem_latency'].mean()
            else:
                tmp_latence = tmp_latence[0]
            tuple2 = row['cellid'] + current_cellid
            tuple3 = row['cellid'] + current_cellid + previous_cellid
            if tuple2 in timetransition2:
                timetransition2[tuple2] += (index - start_index).seconds
                lattransition2[tuple2] += tmp_latence
                numtransition2[tuple2] += 1
            else:
                timetransition2[tuple2] = (index - start_index).seconds
                lattransition2[tuple2] = tmp_latence
                numtransition2[tuple2] = 1
            if tuple3 in timetransition3:
                timetransition3[tuple3] += (index - start_index).seconds
                lattransition3[tuple3] += tmp_latence
                numtransition3[tuple3] += 1
            else:
                timetransition3[tuple3] = (index - start_index).seconds
                lattransition3[tuple3] = tmp_latence
                numtransition3[tuple3] = 1
            tmp_cell = pd.DataFrame([(row['cellid'], current_cellid, (index - start_index).seconds, tmp_latence)],
                                    columns=['cellid', 'previous_cellid', 'dt', 'latence'], index=[index])
            cells2 = cells2.append(tmp_cell)
            start_index = index
            previous_cellid = current_cellid
            current_cellid = row['cellid']

    for item in numtransition2:
        timetransition2[item] /= numtransition2[item]

    for item in numtransition3:
        timetransition3[item] /= numtransition3[item]

    cells2.to_pickle(os.path.join(save_path, "cells2.pkl"))
    np.save(os.path.join(save_path, "timetransitions2.npy"), timetransition2)
    np.save(os.path.join(save_path, "timetransitions3.npy"), timetransition3)

    vars_cat = [var for var in cells2.columns if cells2[var].dtypes == 'O']
    ordinal_enc = OrdinalEncoder()
    cells3 = cells2.copy()
    cells3[vars_cat] = ordinal_enc.fit_transform(cells2[vars_cat])
    X_train, X_test, y_train, y_test = train_test_split(
        cells3.drop('latence', axis=1),  # predictors
        cells3.latence,  # target
        test_size=0.1)  # for reproducibility

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    tree_reg = GradientBoostingRegressor(n_estimators=5000, learning_rate=.001)
    tree_reg.fit(X_train, y_train)

    pred = tree_reg.predict(X_train)
    print('linear train mse: {}'.format(mean_squared_error(y_train, pred)))
    print('linear train rmse: {}'.format(sqrt(mean_squared_error(y_train, pred))))
    print()
    pred = tree_reg.predict(X_test)
    print('linear test mse: {}'.format(mean_squared_error(y_test, pred)))
    print('linear test rmse: {}'.format(sqrt(mean_squared_error(y_test, pred))))

    plt.scatter(y_test, tree_reg.predict(X_test))
    plt.xlabel('True Latency')
    plt.ylabel('Predicted Latency')
    plt.title('Evaluation of Latency Predictions')
    plt.savefig(os.path.join(save_path, "latency_predicition.png"))
    return len(cells2)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Num cell transitions data: ", _prepare_predict_latency("/nfs/ailab/ride/artifacts/modem.csv",
                                                           "/nfs/ailab/ride/artifacts/bandwidth.csv",
                                                           "/nfs/ailab/ride/artifacts/"))
    else:
        print("modem:", sys.argv[1], " bandwidth:", sys.argv[2], " save:",
              sys.argv[2], " Num cells transitions:",
              _prepare_predict_latency(sys.argv[1], sys.argv[2], sys.argv[3]))
