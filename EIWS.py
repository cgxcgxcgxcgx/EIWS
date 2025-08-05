import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# ========== 数据加载函数 ==========
def load_data_from_csv(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values  # 前7列为特征
    y = df.iloc[:, -1].values   # 最后一列为目标变量（波速）
    return X, y

# ========== 构建神经网络模型 ==========
def build_nn_model(lr):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(515, activation='relu', input_shape=(7,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse', metrics=['mae'])
    return model

# ========== Step 1: 初始训练 ==========
X, y = load_data_from_csv('original_data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_train_preds = rf_model.predict(X_train)
rf_test_preds = rf_model.predict(X_test)

# 神经网络训练（learning rate = 0.001）
nn_model = build_nn_model(lr=0.001)
nn_model.fit(X_train, y_train, epochs=1000, batch_size=16, verbose=1)
nn_model.save_weights('pretrained_weights.h5')

nn_train_preds = nn_model.predict(X_train).flatten()
nn_test_preds = nn_model.predict(X_test).flatten()

# 堆叠模型 + SVR
stacked_train = np.vstack((rf_train_preds, nn_train_preds)).T
stacked_test = np.vstack((rf_test_preds, nn_test_preds)).T

meta_model = SVR(kernel='rbf', C=1000, epsilon=0.01)
meta_model.fit(stacked_train, y_train)
stacked_test_preds = meta_model.predict(stacked_test)

# 评估初始模型
print("Initial Random Forest Test MSE:", mean_squared_error(y_test, rf_test_preds))
print("Initial Neural Network Test MSE:", mean_squared_error(y_test, nn_test_preds))
print("Initial Stacking Model Test MSE:", mean_squared_error(y_test, stacked_test_preds))

# ========== Step 2: 迁移学习 ==========
X_new, y_new = load_data_from_csv('transfer_data.csv')
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, test_size=0.2, random_state=24)

nn_transfer_model = build_nn_model(lr=0.0001)
nn_transfer_model.load_weights('pretrained_weights.h5')

nn_transfer_model.fit(X_new_train, y_new_train, epochs=300, batch_size=16, verbose=1)
nn_new_preds = nn_transfer_model.predict(X_new_test).flatten()

print("Transferred NN Test MSE:", mean_squared_error(y_new_test, nn_new_preds))
