import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras import regularizers
import matplotlib
import matplotlib.pyplot as plt
from keras.regularizers import l1
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from pandas.plotting import table
from sklearn.preprocessing import MinMaxScaler


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def parse_ml100k():
    # Read data
    ds = pd.read_csv('ml-100k/u.data', sep='\t',
                     names=['user_id', 'movie_id', 'rating', 'unix_timestamp'], encoding='latin-1')

    # Dropping the columns that are not required
    ds.drop("unix_timestamp", inplace=True, axis=1)

    # Generate matrix
    ds = ds.pivot_table(index=['user_id', ], columns=['movie_id'],
                        values='rating').reset_index(drop=True)

    return ds


def preprocessing(Y: np.array):
    # Filling
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    Y = imp.fit_transform(Y.transpose()).transpose()

    # Scaling, Normalization
    Y = MinMaxScaler().fit_transform(X=Y)

    return Y


# Train n Test
def tnt(X, Y, h, epochs, lr=0.01, m=0., rg=None):
    # Split data
    kfold = KFold(n_splits=5, shuffle=True)

    rmseList = []
    maeList = []

    # Build model
    model = Sequential()
    model.add(Dense(h, input_shape=(943,), activation='sigmoid',
                    kernel_regularizer=rg, activity_regularizer=rg))
    model.add(Dense(1682, activation='linear',
                    kernel_regularizer=rg, activity_regularizer=rg))

    sgd = SGD(lr=lr, momentum=m, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd,
                  metrics=[rmse, 'mae'])

    # Train model
    for i, (train, test) in enumerate(kfold.split(X)):
        # es = EarlyStopping(monitor='val_loss', mode='min')
        # history = model.fit(X[train], Y[train], epochs=epochs,
        #             batch_size=10, verbose=0, validation_data=(X[test], Y[test]), callbacks=[es])
        history = model.fit(X[train], Y[train], epochs=epochs,
                            batch_size=10, verbose=0, validation_data=(X[test], Y[test]))

        # Evaluate model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        rmseList.append(scores[0])
        maeList.append(scores[1])

    # Return average
    return [np.mean(rmseList), np.mean(maeList)], history


def save_df(df, title):
    print(df.to_markdown())
    text_file = open("./tables/{}.md".format(title), "w")
    text_file.write(df.to_markdown())
    text_file.close()


# Plot helper
def save_plot(pss, xlabel, ylabel, title, legend):
    for ps in pss:
        plt.plot(ps)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.title("{}".format(title))
    plt.legend(legend)
    plt.savefig("./images/{}.png".format(title), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1", help="run a1", action="store_true")
    parser.add_argument("--a2", help="run a2",
                        action="store_true")
    parser.add_argument("--a3", help="run a3", action="store_true")
    parser.add_argument("--a4", help="run a4", action="store_true")
    args = parser.parse_args()

    # Parse dataset, keep only Y
    ml100k = parse_ml100k().to_numpy()

    # A1
    if args.a1:
        ml100k_mr = np.around(np.nanmean(ml100k, axis=1), 1)

        df = pd.DataFrame({'x': ml100k_mr})
        df['x'].hist(bins=11)
        plt.title("Mean rating density")
        plt.savefig("./images/mean-rating-histogram.png".format(),
                    bbox_inches='tight')
        plt.close()

        print("Ml100k mean of mean ratings : {}".format(
            np.mean(ml100k_mr).astype(str)))
        print("Ml100k covar of mean rating : {}".format(
            np.cov(ml100k_mr).astype(str)))
        print("Ml100k range of mean rating : {}".format(
            ((np.nanmax(ml100k_mr)) - np.nanmin(ml100k_mr)).astype(str)))

    # Generate X and Y, X in One-hot encoded form
    XY = (np.eye(943), preprocessing(ml100k))

    # A2.ζ
    if args.a2:
        data = []
        for epochs in (20, 50, 100):
            for h in (10, 15, 20, 25):
                me, history = tnt(*XY, h, epochs=epochs)
                save_plot([history.history['loss'], history.history['val_loss']], 'epochs',
                          'loss', 'e={}h={}'.format(epochs, h), ['Loss', 'Validation loss'])
                data.append((epochs, h, *me))
        df = pd.DataFrame(data, columns=['Epochs', 'H', 'RMSE', 'MAE'])
        save_df(df, 'A2.ζ')

    h_opt = 20

    # Α3 Find optimal lr, m
    if args.a3:
        data = []
        for epochs in (20, 50, 100):
            for lr, m in [(0.001, 0.2), (0.001, 0.6), (0.05, 0.6), (0.1, 0.6)]:
                me, history = tnt(*XY, h_opt, epochs=epochs, lr=lr, m=m)
                save_plot([history.history['loss'], history.history['val_loss']], 'epochs',
                          'loss', 'e={}lr={}m={}'.format(epochs, lr, m), ['Loss', 'Validation loss'])
                data.append((epochs, lr, m, *me))
        df = pd.DataFrame(data, columns=['Epochs', 'lr', 'm', 'RMSE', 'MAE'])
        save_df(df, 'A3')

    opt_lr, opt_m = 0.05, 0.6

    # A4
    if args.a4:
        data = []
        for epochs in (20, 50, 100, 300):
            for r in (0.1, 0.5, 0.9):
                me, history = tnt(
                    *XY, h_opt, epochs=epochs, lr=opt_lr, m=opt_m, rg=l1(l=r))
                save_plot([history.history['loss'], history.history['val_loss']], 'epochs',
                          'loss', 'e={}r={}'.format(epochs, r), ['Loss', 'Validation loss'])
                data.append((epochs, r, *me))
        df = pd.DataFrame(data, columns=['Epochs', 'r', 'RMSE', 'MAE'])
        save_df(df, 'A4')


if __name__ == '__main__':
    main()
