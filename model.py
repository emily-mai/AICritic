import pandas as pd
import numpy as np
import tensorflow_hub as hub
import asyncio
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split as tts


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
encoder_model = hub.load(module_url)
print("module %s loaded" % module_url)


def load_data():
    plots_raw = pd.read_csv('data/all_data.csv')
    ratings_raw = pd.read_csv('data/IMDb ratings.csv')
    plots_raw = plots_raw[['imdb_id', 'plot_synopsis']]
    plots_raw = plots_raw.set_index('imdb_id')
    ratings_raw = ratings_raw[['imdb_title_id', 'weighted_average_vote']]
    ratings_raw = ratings_raw.set_index('imdb_title_id')
    raw = plots_raw.join(ratings_raw).dropna()
    return raw


def background(f):
    def wrapped(*args):
        return asyncio.get_event_loop().run_in_executor(None, f, *args)
    return wrapped


@background
def embed_text_data(data, column, pair):
    messages = data[column].values[pair[0]:pair[1]]
    embeddings = encoder_model(messages)
    embeddings_df = pd.DataFrame(embeddings.numpy())
    embeddings_df.to_csv("embeddings{}.csv".format(pair[1]))
    print(embeddings_df.head)
    print('function finished for ' + str(pair))


def embed_text_parallel(data, column, load_only):
    # Embed text data with universal sentence encoder
    indices = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6000),
               (6000, 7000), (7000, 8000), (8000, 9000), (9000, 10000), (10000, 11707)]
    if not load_only:
        for pair in indices:
            embed_text_data(data, column, pair)
        print("loop finished")

    dataframe = pd.DataFrame()
    for pair in indices:
        df = pd.read_csv("embeddings{}.csv".format(pair[1]))
        dataframe = pd.concat([dataframe, df], ignore_index=True)
    dataframe = dataframe.drop(dataframe.iloc[:, 0:1], axis=1)
    print(dataframe.head)
    dataframe.to_csv("data/embeddings_all.csv")
    return dataframe


def partition_data(embeddings, data_raw):
    embeddings = embeddings.drop(embeddings.iloc[:, 0:1], axis=1)
    embeddings['imdb_id'] = data_raw.index
    y = data_raw['weighted_average_vote']
    X_train, X_test, y_train, y_test = tts(embeddings, y, test_size=0.1, random_state=0)
    output = pd.DataFrame()
    output['imdb_id'] = X_test['imdb_id'].reset_index(drop=True)
    X_train = X_train.drop(columns=['imdb_id'])
    X_test = X_test.drop(columns=['imdb_id'])
    return X_train, X_test, y_train, y_test, output


def evaluate_model(embed=False, fit=False):
    # data loading and preprocessing
    data_raw = load_data()
    print(data_raw.head)
    if embed:
        X = embed_text_parallel(data_raw, 'plot_synopsis', load_only=True)
    else:
        X = pd.read_csv('data/embeddings_all.csv')
    X_train, X_test, y_train, y_test, output = partition_data(X, data_raw)

    # save checkpoints
    checkpoint_name = 'weights.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    # train the model
    NN_model = model_cnn(input_dim=len(X_train.columns))
    if fit:
        NN_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, callbacks=callbacks_list)

    # load weights of best model
    weights_file = 'weights.hdf5'  # choose the best checkpoint
    NN_model.load_weights(weights_file)  # load it
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    # test the model
    output['prediction'] = NN_model.predict(X_test)
    output['y_test'] = np.asarray(y_test).reshape(-1, 1)
    output['error'] = (output['y_test'].subtract(output['prediction'])).abs()
    MAE = mean_absolute_error(output['y_test'], output['prediction'])
    output = output[output['error'] < 2]
    error_no_outlier = output['error'].mean()
    print("mean absolute error without outliers: {}".format(error_no_outlier))
    print("mean absolute error: {}".format(MAE))


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
    # model = Sequential()
    # model.add(Embedding(vocab_size, input_dim, input_length=input_dim))
    # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
    # model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D())
    # model.add(Flatten())
    # model.add(Dense(250, activation='relu'))
    #
    # # The Output Layer :
    # model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    #
    # # Compile the network :
    # model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    # model.summary()

    int_sequences_input = Input(shape=(None,), dtype="int64")
    embedding_layer = Embedding(vocab_size, input_dim, trainable=False, input_length=input_dim)
    embedded_sequences = embedding_layer(int_sequences_input)
    x = Conv1D(128, 5, activation="relu")(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation="relu")(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation="relu")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    preds = Dense(1, kernel_initializer='normal', activation="linear")(x)
    model = Model(int_sequences_input, preds)
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.summary()
    return model


def predict_plot_rating(text):
    embedding = encoder_model([text]).numpy()
    model = model_nn(input_dim=embedding.shape[1])
    model.load_weights("weights.hdf5")
    prediction = model.predict(embedding)[0][0]
    return prediction
