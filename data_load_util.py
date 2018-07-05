from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add

# Load document
def load_doc(filename):

    # Read only mode
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# Load Dataset
def load_set(filename):

    doc = load_doc(filename)
    dataset = list()

    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue

        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# Load Cleaned Descriptions
def load_clean_descriptions(filename, dataset):

    doc = load_doc(filename)
    descriptions = dict()

    for line in doc.split('\n'):

        # Split by whitespace
        tokens = line.split()

        # Split ID and Description
        image_id, image_desc = tokens[0], tokens[1:]

        # Skip images if they don't belong to the dataset
        if image_id in dataset:
            # Create list
            if image_id not in descriptions:
                descriptions[image_id] = list()

            # Wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'

            # Store
            descriptions[image_id].append(desc)

    return descriptions


# Load Photo Features
def load_photo_features(filename, dataset):
    # Load All
    all_features = load(open(filename, 'rb'))

    # Filter
    features = {k: all_features[k] for k in dataset}

    return features


# Description dictionary to List
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# Fit KERAS tokenizer
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# Max Length of Description with most words
def get_max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# Create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()

    # Iterate through every image identifier
    for key, desc_list in descriptions.items():

        # Iterate through each description for the image
        for desc in desc_list:
            # Encode
            seq = tokenizer.texts_to_sequences([desc])[0]

            # Split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):

                # Split into I/O pair
                in_seq, out_seq = seq[:i], seq[i]

                # Pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                # Encode
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                # Store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)

    return array(X1), array(X2), array(y)


# Define Model
def define_model(vocab_size, max_length):
    # Feature Extractor
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence Model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)

    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Combine [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# Load Training Set
def load_training_set():
    print('\nLoading Train Set\n')

    # load training dataset (6K)
    filename = 'data/Flickr8k_text/Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print('Dataset:\t' + str(len(train)))

    # Descriptions
    train_descriptions = load_clean_descriptions('data/descriptions.txt', train)
    print('Descriptions (Train):\t' + str(len(train_descriptions)))

    # Photo features
    train_features = load_photo_features('data/features.pkl', train)
    print('Photos (Train):\t' + str(len(train_features)))

    # Prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size:\t' + str(vocab_size))

    # Get maximum sequence length
    max_length = get_max_length(train_descriptions)
    print('Description Length:\t' + str(max_length))

    # Prepare sequences
    X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size=vocab_size)

    return X1train, X2train, ytrain, vocab_size, max_length, tokenizer


# Load Test Set
def load_test_set(vocab_size, max_length, tokenizer):

    print('\nLoading Test Set\n')

    # Load Test set
    filename = 'data/Flickr8k_text/Flickr_8k.devImages.txt'
    test = load_set(filename)
    print('Dataset:\t' + str(len(test)))

    # Descriptions
    test_descriptions = load_clean_descriptions('data/descriptions.txt', test)
    print('Descriptions (Test):\t' + str(len(test_descriptions)))

    # Photo features
    test_features = load_photo_features('data/features.pkl', test)
    print('Photos (Test):\t' + str(len(test_features)))

    # # Prepare tokenizer
    # tokenizer = create_tokenizer(test_descriptions)
    # vocab_size = len(tokenizer.word_index) + 1
    # print('Vocabulary Size:\t' + str(vocab_size))
    #
    # # Get maximum sequence length
    # max_length = get_max_length(test_descriptions)
    # print('Description Length:\t' + str(max_length))

    # Prepare sequences
    X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size=vocab_size)

    return X1test, X2test, ytest


# Init Data Function
def init_data_load():
    print('\nData Load Initialized\n')
    X1train, X2train, ytrain, vocab_size, max_length, tokenizer = load_training_set()
    X1test, X2test, ytest = load_test_set(vocab_size, max_length, tokenizer)

    print('\nData Load Ended\n')

    return X1train, X2train, ytrain, vocab_size, max_length, tokenizer, X1test, X2test, ytest


if __name__ == "__main__":
    init_data_load()
