# Tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam  # and any other optimizers you might use
# My functions:
from src.utils.CRPS import *  # CRPS metrics




def build_emb_model(
    n_features,
    n_outputs,
    hidden_nodes,
    emb_size,
    max_id,
    compile=False,
    optimizer="Adam",
    lr=0.01,
    loss=crps_cost_function,
    activation="relu",
    reg=None,
):
    """

    Args:
        n_features: Number of features
        n_outputs: Number of outputs
        hidden_nodes: int or list of hidden nodes
        emb_size: Embedding size
        max_id: Max embedding ID
        compile: If true, compile model
        optimizer: Name of optimizer
        lr: learning rate
        loss: loss function
        activation: Activation function for hidden layer

    Returns:
        model: Keras model
    """
    if type(hidden_nodes) is not list:
        hidden_nodes = [hidden_nodes]

    features_in = Input(shape=(n_features,))
    id_in = Input(shape=(1,))
    emb = Embedding(max_id + 1, emb_size)(id_in)
    emb = Flatten()(emb)
    x = Concatenate()([features_in, emb])
    for h in hidden_nodes:
        x = Dense(h, activation=activation, kernel_regularizer=reg)(x)
    x = Dense(n_outputs, activation="linear", kernel_regularizer=reg)(x)
    model = Model(inputs=[features_in, id_in], outputs=x)

    if compile:
        opt = keras.optimizers.__dict__[optimizer](learning_rate=lr)
        model.compile(optimizer=opt, loss=loss)
    return model