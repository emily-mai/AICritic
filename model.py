from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.layers import Dense, Embedding, Flatten, Input, Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_absolute_error


def model_nn(input_dim):
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(128, kernel_initializer='normal', input_dim=input_dim, activation='relu'))
    NN_model.add(Dropout(0.3))

    # The Hidden Layers :
    NN_model.add(Dense(512, activation='relu'))
    NN_model.add(Dropout(0.3))
    NN_model.add(Dense(256, activation='relu'))
    NN_model.add(Dropout(0.3))
    NN_model.add(Dense(128, activation='relu'))
    NN_model.add(Dropout(0.3))

    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    NN_model.summary()
    return NN_model


def model_cnn(input_dim, vocab_size=5000):
    model = Sequential()
    model.add(Embedding(vocab_size, input_dim, input_length=input_dim))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))

    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network :
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.summary()
    return model


def error(predictions, y_test):
    MAE = mean_absolute_error(y_test, predictions)
    return MAE
