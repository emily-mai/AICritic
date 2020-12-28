import pandas as pd
import numpy as np
import tensorflow_hub as hub
import asyncio
import model
from sklearn.model_selection import train_test_split as tts
from keras.callbacks import ModelCheckpoint


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
def embed_text(data, column, pair, encoder_model):
    messages = data[column].values[pair[0]:pair[1]]
    embeddings = encoder_model(messages)
    embeddings_df = pd.DataFrame(embeddings.numpy())
    embeddings_df.to_csv("embeddings{}.csv".format(pair[1]))
    print(embeddings_df.head)
    print('function finished for ' + str(pair))


def embed_text_parallel(data, column, load_only):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    encoder_model = hub.load(module_url)
    print("module %s loaded" % module_url)
    # Embed text data with universal sentence encoder
    indices = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6000),
               (6000, 7000), (7000, 8000), (8000, 9000), (9000, 10000), (10000, 11707)]
    if not load_only:
        for pair in indices:
            embed_text(data, column, pair, encoder_model)
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


if __name__ == '__main__':
    # data loading and preprocessing
    data_raw = load_data()
    print(data_raw.head)
    # embeddings = embed_text_parallel(data_raw, 'plot_synopsis', load_only=True)
    X = pd.read_csv('data/embeddings_all.csv')
    X_train, X_test, y_train, y_test, output = partition_data(X, data_raw)

    # save checkpoints
    checkpoint_name = 'weights.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    # train the model
    NN_model = model.model_cnn(input_dim=len(X_train.columns))
    NN_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, callbacks=callbacks_list)

    # load weights of best model
    weights_file = 'weights.hdf5'  # choose the best checkpoint
    NN_model.load_weights(weights_file)  # load it
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    # test the model
    output['prediction'] = NN_model.predict(X_test)
    output['y_test'] = np.asarray(y_test).reshape(-1, 1)
    output['error'] = (output['y_test'].subtract(output['prediction'])).abs()
    error = model.error(output['prediction'], output['y_test'])
    print("mean absolute error: {}".format(error))
