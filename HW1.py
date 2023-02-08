import torch
from torch import nn
from torch.nn import functional as F
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Input, layers
import pandas as pd
from d2l import tensorflow as d2l


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#scaling image values
train_images = train_images / 255
test_images = test_images / 255
print(train_images.shape)

x_val = train_images[-10000:]
y_val = train_labels[-10000:]

x_train = train_images[:-10000]
y_train = train_labels[:-10000]


#My Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu',
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(128, activation='relu', 
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(64, activation='relu', 
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(10) ])

#Compile the model
model.compile(optimizer='SGD', 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

#Train Model
history = model.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))

model.save_weights('BaseModel.params')

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Results')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')
plt.show()


#******************************88
#Model 2 with
model_weightdecay = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu', 
        kernel_regularizer='l2', 
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(128, activation='relu', 
        kernel_regularizer='l2', 
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(64, activation='relu', 
        kernel_regularizer='l2', 
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(10) ])

model_weightdecay.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

history = model_weightdecay.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.title('Model with weight decay Results')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')
plt.show()

#******************************88
#Model 3 with dropout
model_dropout = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu', 
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu', 
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu', 
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10) ])

model_dropout.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

history = model_dropout.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.title('Model with Dropout Results')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')
plt.show()


#********************************************************************
#Model 2(Weight Decay) with preloaded weights from saved base model

model_weightdecay.load_weights('BaseModel.params')

model_weightdecay.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

history = model_weightdecay.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.title('Model with weight decay Results')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')
plt.show()

#************************************************************************
#Model 3 (Dropout) with preloaded weights from saved base model
model_dropout.load_weights('BaseModel.params')

model_dropout.compile(optimizer='SGD',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

history = model_dropout.fit(x_train, y_train, batch_size=16, epochs=20, validation_data=(x_val, y_val))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.title('Model with Dropout Results')
plt.xlabel('epochs')
plt.legend(['train_acc', 'val_acc', 'train_loss'], loc = 'upper left')
plt.show()

#PART 2
#************************************************************************
#Code from book: for reference on my own model

class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))

data = KaggleHouse(batch_size=32)
print(data.raw_train.shape)
print(data.raw_val.shape)

@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding.
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()

data.preprocess()
print(data.train.shape)


@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: tf.constant(x.values, dtype=tf.float32)
    # Logarithm of prices
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               tf.reshape(tf.math.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)

def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),
                                data.train.loc[idx]))
    return rets

def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models


trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
plt.title("Kaggle Regression Model from Class")
plt.xlabel("epochs")
plt.show() #Shows plot of original model that we used in class

#**********************************************************************************88
#My more complex defined model uses a fully connected netwoork with 3 hidden layers
class MyModel(d2l.Module):  #@save
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.net = tf.keras.Sequential([
                tf.keras.layers.Dense(256, kernel_initializer=initializer, activation='relu'),
                tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='relu'),
                tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu'),
                tf.keras.layers.Dense(1, kernel_initializer=initializer)])
    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)

    def configure_optimizers(self):
        return tf.keras.optimizers.SGD(self.lr)

#new k-fold function calling my new model
def new_k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = MyModel(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models

trainer = d2l.Trainer(max_epochs=10)
models = new_k_fold(trainer, data, k=5, lr=0.01)
plt.title("My Improved Regression Model")
plt.xlabel("epochs")
plt.show() #plot of my models performance

#*******************************************************************************************
#Trying the regression with a model using weight decay.

class MyModelWeightDecay(d2l.Module):  #@save
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.net = tf.keras.Sequential([
                tf.keras.layers.Dense(256, kernel_regularizer='l2', kernel_initializer=initializer, activation='relu'),
                tf.keras.layers.Dense(128, kernel_regularizer='l2', kernel_initializer=initializer, activation='relu'),
                tf.keras.layers.Dense(64, kernel_regularizer='l2', kernel_initializer=initializer, activation='relu'),
                tf.keras.layers.Dense(1, kernel_initializer=initializer)])
    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)

    def configure_optimizers(self):
        return tf.keras.optimizers.SGD(self.lr)

#new k-fold function calling my new model
def new_k_fold_WeightDecay(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = MyModelWeightDecay(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models

trainer = d2l.Trainer(max_epochs=10)
models = new_k_fold_WeightDecay(trainer, data, k=5, lr=0.1)
plt.title("My Regression Model with Wight Decay")
plt.xlabel("epochs")
plt.show()

#*******************************************************************************************
#Trying the regression with a model using dropout.

class MyModelDropout(d2l.Module):  #@save
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.net = tf.keras.Sequential([
                tf.keras.layers.Dense(256, kernel_initializer=initializer, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(128, kernel_initializer=initializer, activation='relu'),
                tf.keras.layers.Dropout(0,3),
                tf.keras.layers.Dense(64, kernel_initializer=initializer, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, kernel_initializer=initializer)])
    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)

    def configure_optimizers(self):
        return tf.keras.optimizers.SGD(self.lr)

#new k-fold function calling my new model
def new_k_fold_Dropout(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = MyModelDropout(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models

trainer = d2l.Trainer(max_epochs=10)
models = new_k_fold_Dropout(trainer, data, k=5, lr=0.1)
plt.title("My Regression Model with Dropout")
plt.xlabel("epochs")
plt.show()





