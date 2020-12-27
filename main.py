import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import asyncio
from sklearn.model_selection import train_test_split as tts
tf.executing_eagerly()

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)


def load_data():
    plots_raw = pd.read_csv('data/all_data.csv')
    ratings_raw = pd.read_csv('data/IMDb ratings.csv')
    plots_raw = plots_raw[['imdb_id', 'plot_synopsis']]
    plots_raw = plots_raw.set_index('imdb_id')
    ratings_raw = ratings_raw[['imdb_title_id', 'weighted_average_vote']]
    ratings_raw = ratings_raw.set_index('imdb_title_id')
    raw = plots_raw.join(ratings_raw).dropna()
    # print(data_raw.head())
    return raw


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args)
    return wrapped


@background
def embed_text(data, column, pair):
    messages = data[column].values[pair[0]:pair[1]]
    embeddings = model(messages)
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
            embed_text(data, column, pair)
        print("loop finished")

    dataframe = pd.DataFrame()
    for pair in indices:
        df = pd.read_csv("embeddings{}.csv".format(pair[1]))
        dataframe = pd.concat([dataframe, df], ignore_index=True)
    dataframe = dataframe.drop(dataframe.iloc[:, 0:1], axis=1)
    print(dataframe.head)
    dataframe.to_csv("embeddings_all")
    return dataframe


if __name__ == '__main__':
    data_raw = load_data()
    print(data_raw.head)
    embeddings = embed_text_parallel(data_raw, 'plot_synopsis', load_only=True)
    y = data_raw['weighted_average_vote']
    X_train, X_test, y_train, y_test = tts(embeddings, y, test_size=0.2, random_state=0)
    print(len(X_train), len(y_train), len(X_test), len(y_test))
