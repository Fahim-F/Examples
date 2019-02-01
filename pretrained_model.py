import tensorflow as tf
# Don't use all GPU Memory available
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#
import os
import time
import json
import argparse
import myNet
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.datasets import cifar10
from keras import optimizers
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50
import pdb

def run_cifar10(batch_size,
                nb_epoch,
                dropout_rate,
                learning_rate,
                weight_decay):
    """ Run CIFAR10 experiments

    :param batch_size: int -- batch size
    :param nb_epoch: int -- number of training epochs
    :param depth: int -- network depth
    :param nb_dense_block: int -- number of dense blocks
    :param nb_filter: int -- initial number of conv filter
    :param growth_rate: int -- number of new filters added by conv layers
    :param dropout_rate: float -- dropout rate
    :param learning_rate: float -- learning rate
    :param weight_decay: float -- weight decay
   
    """

    ###################
    # Data processing #
    ###################

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    nb_classes = len(np.unique(y_train))
    img_dim = X_train.shape[1:]

    if K.image_data_format() == "channels_first":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalisation
    X = np.vstack((X_train, X_test))
    # 2 cases depending on the image ordering
    if K.image_data_format() == "channels_first":
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

    elif K.image_data_format() == "channels_last":
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std

    ###################
    # Construct model #
    ###################
    # (1) Create model
    # use available model or create your model 
    # Available model: https://keras.io/applications/#available-models
    
    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    
    # let's add a fully-connected layer
    
    # and a logistic layer -- let's say we have nb_classes
    

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False
        
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)
    
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 170 layers and unfreeze the rest:

    

    model.summary()

    # (2) Build optimizer
    opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #opt = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    
    # (3) Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=["accuracy"])

    ####################
    # Network training #
    ####################

    print("Training")

    list_train_loss = []
    list_test_loss = []
    list_learning_rate = []

    for e in range(nb_epoch):

        if e == int(0.5 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

        if e == int(0.75 * nb_epoch):
            K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))

        split_size = batch_size
        num_splits = X_train.shape[0] / split_size
        arr_splits = np.array_split(np.arange(X_train.shape[0]), num_splits)

        l_train_loss = []
        start = time.time()
        #pdb.set_trace()
        for batch_idx in arr_splits:

            X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]
            train_logloss, train_acc = model.train_on_batch(X_batch, Y_batch)

            l_train_loss.append([train_logloss, train_acc])
            #print(batch_idx)

        test_logloss, test_acc = model.evaluate(X_test,
                                                Y_test,
                                                verbose=0,
                                                batch_size=64)
        list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
        list_test_loss.append([test_logloss, test_acc])
        list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
        # to convert numpy array to json serializable
        print('Epoch %s/%s, Time: %s , test_accuracy: %s' % (e + 1, nb_epoch, time.time() - start , test_acc))
        
        d_log = {}
        d_log["batch_size"] = batch_size
        d_log["nb_epoch"] = nb_epoch
        d_log["optimizer"] = opt.get_config()
        d_log["train_loss"] = list_train_loss
        d_log["test_loss"] = list_test_loss
        d_log["learning_rate"] = list_learning_rate

        json_file = os.path.join('./log/experiment_log_cifar10.json')
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run CIFAR10 experiment')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=10, type=int,
                        help='Number of epochs')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4,
                        help='L2 regularization on weights')
    

    args = parser.parse_args()

    print("Network configuration:")
    for name, value in parser.parse_args()._get_kwargs():
        print(name, value)

    list_dir = ["./log", "./figures"]
    for d in list_dir:
        if not os.path.exists(d):
            os.makedirs(d)

    run_cifar10(args.batch_size,
                args.nb_epoch,
                args.dropout_rate,
                args.learning_rate,
                args.weight_decay)