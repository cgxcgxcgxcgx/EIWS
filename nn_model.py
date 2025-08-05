import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def build_nn_model(lr):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(7,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse', metrics=['mae'])
    return model

def train_nn(X_train, y_train, lr, epochs, save_path):
    model = build_nn_model(lr)
    early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=1, callbacks=[early_stop])
    model.save(save_path)
    plot_loss(history.history['loss'], save_path.replace('.h5', '_loss.png'))
    return model

def fine_tune_nn(pretrained_path, X_finetune, y_finetune, lr, epochs, save_path):
    model = load_model(pretrained_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    history = model.fit(X_finetune, y_finetune, epochs=epochs, batch_size=16, verbose=1, callbacks=[early_stop])
    model.save(save_path)
    plot_loss(history.history['loss'], save_path.replace('.h5', '_loss.png'))
    return model

def load_nn_model(path):
    return load_model(path)

def plot_loss(loss, save_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
