from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump, load
from tensorflow.python.keras import models
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from src import datasets
from src import infer
from src import preprocessing

def train():
    '''Trains a separable cnn model.

    '''

    # The following hyperparameter values are based on Tensorflow text classification recommended values
    # https://developers.google.com/machine-learning/guides/text-classification/step-4
    # https://developers.google.com/machine-learning/guides/text-classification/step-5

    # Tokenization/Vectorization parameters
    top_k = 20000 # Tokenize/Vectorize top k words from training corpus
    max_sequence_length = 25 # Maximum number of words per sample

    # Model parameters
    embedding_dim = 200 # Word embedding dimensions (50 - 300)
    blocks = 2
    filters = 64
    kernel_size = 3
    dropout_rate = 0.2
    learning_rate = 1e-3
    epochs = 1000
    batch_size = 128
    pool_size = 3
    num_classes = 4 # TODO: Compute number of classes

    x_train, y_train, x_test, y_test = datasets.get_train_test_data()
    tokenizer = text.Tokenizer(num_words=top_k)
    tokenizer.fit_on_texts(x_train)
    dump(tokenizer, 'models/cnn/tokenizer.joblib')

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    max_length = len(max(x_train, key=len))
    if max_length > max_sequence_length:
        max_length = max_sequence_length
    
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_length)

    # Create model instance.
    model = sepcnn_model(
        blocks=blocks,
        filters=filters,
        kernel_size=kernel_size,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        pool_size=pool_size,
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
        num_features=min(len(tokenizer.word_index) + 1, top_k))

    # Compile model with learning parameters.
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2),
        tf.keras.callbacks.TensorBoard(log_dir=logdir)]

    # Train and validate model.
    history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_test, y_test),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('models/cnn/model.h5')
    #return history['val_acc'][-1], history['val_loss'][-1]

def eval(filename):
    ''' Evaluates the tensorflow cnn model.

    Args:
        input (filename): The filename to evaluate.
    '''
    tokenizer = load('models/cnn/tokenizer.joblib')
    x = preprocessing.process_filename(filename)
    x = tokenizer.texts_to_sequences([x])
    x = sequence.pad_sequences(x, 25)
    model = tf.keras.models.load_model('models/cnn/model.h5')
    y = model.predict_proba(x)
    label, confidence = infer.get_label(y)
    print('Predicted class \'{label}\' with {confidence:.2f}% confidence.'
        .format(label=label, confidence=confidence*100))

def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_classes,
                 num_features):
    """Creates an instance of a separable CNN model.

    # Arguments
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of the layers.
        kernel_size: int, length of the convolution window.
        embedding_dim: int, dimension of the embedding vectors.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.
        num_features: int, number of words (embedding input dimension).

    # Returns
        A sepCNN model instance.
    """
    op_activation = 'sigmoid'
    op_units = 1
    if num_classes > 2:
        op_activation = 'softmax'
        op_units = num_classes
    
    model = models.Sequential()

    # Add embedding layer.
    model.add(Embedding(input_dim=num_features,
                        output_dim=embedding_dim,
                        input_length=input_shape[0]))

    for _ in range(blocks-1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
    return model