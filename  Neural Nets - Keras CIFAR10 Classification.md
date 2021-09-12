# Imports


```python
from numpy.random import seed
seed(888)
from tensorflow import set_random_seed
set_random_seed(404)
```


```python
import os
import numpy as np
import tensorflow as tf
import itertools

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from IPython.display import display
from keras.preprocessing.image import array_to_img
from keras.callbacks import TensorBoard

from time import strftime

from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt

%matplotlib inline
```

# Constants


```python
LOG_DIR = 'tensorboard_cifar_logs/'

LABEL_NAMES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
COLOR_CHANNELS = 3
TOTAL_INPUTS = IMAGE_PIXELS * COLOR_CHANNELS
NR_CLASSES = 10

VALIDATION_SIZE = 10000
SMALL_TRAIN_SIZE = 1000
```

# Get the Data


```python
(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
```

# Preprocess Data


```python
x_train_all, x_test = x_train_all / 255.0, x_test / 255.0
```


```python
x_train_all[0][0][0][0]
```


```python
x_train_all = x_train_all.reshape(x_train_all.shape[0], TOTAL_INPUTS)
```


```python
x_train_all.shape
```


```python
x_test = x_test.reshape(len(x_test), TOTAL_INPUTS)
print(f'Shape of x_test is {x_test.shape}')
```

### Create Validation Dataset


```python
x_val = x_train_all[:VALIDATION_SIZE]
y_val = y_train_all[:VALIDATION_SIZE]
x_val.shape
```

#### Create two numpy arrays ```x_train``` and ```y_train``` that have the shape (40000, 3072) and (40000, 1) respectively. 
#### They contain the last 40000 values from ```x_train_all``` and ```y_train_all``` respectively. 


```python
x_train = x_train_all[VALIDATION_SIZE:]
y_train = y_train_all[VALIDATION_SIZE:]
x_train.shape
```

### Create a small dataset (for illustration)


```python
x_train_xs = x_train[:SMALL_TRAIN_SIZE]
y_train_xs = y_train[:SMALL_TRAIN_SIZE]
```

# Define the Neural Network using Keras


```python
model_1 = Sequential([
    Dense(units=128, input_dim=TOTAL_INPUTS, activation='relu', name='m1_hidden1'),
    Dense(units=64, activation='relu', name='m1_hidden2'),
    Dense(16, activation='relu', name='m1_hidden3'),
    Dense(10, activation='softmax', name='m1_output')
])

model_1.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
```


```python
model_2 = Sequential()
model_2.add(Dropout(0.2, seed=42, input_shape=(TOTAL_INPUTS,)))
model_2.add(Dense(128, activation='relu', name='m2_hidden1'))
model_2.add(Dense(64, activation='relu', name='m2_hidden2'))
model_2.add(Dense(15, activation='relu', name='m2_hidden3'))
model_2.add(Dense(10, activation='softmax', name='m2_output'))

model_2.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
```

#### Create a third model, ```model_3``` that has two dropout layers. 
#### The second dropout layer added after the first hidden layer and have a dropout rate of 25%. 


```python
model_3 = Sequential()
model_3.add(Dropout(0.2, seed=42, input_shape=(TOTAL_INPUTS,)))
model_3.add(Dense(128, activation='relu', name='m3_hidden1'))
model_3.add(Dropout(0.25, seed=42))
model_3.add(Dense(64, activation='relu', name='m3_hidden2'))
model_3.add(Dense(15, activation='relu', name='m3_hidden3'))
model_3.add(Dense(10, activation='softmax', name='m3_output'))

model_3.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
```


```python
type(model_1)
```


```python
model_1.summary()
```


```python
32*32*3*128 + 128 + (128*64 + 64) + (64*16 + 16) + (16*10 + 10)
```

# Tensorboard (visualising learning)


```python
def get_tensorboard(model_name):

    folder_name = f'{model_name} at {strftime("%H %M")}'
    dir_paths = os.path.join(LOG_DIR, folder_name)

    try:
        os.makedirs(dir_paths)
    except OSError as err:
        print(err.strerror)
    else:
        print('Successfully created directory')

    return TensorBoard(log_dir=dir_paths)
```

# Fit the Model


```python
samples_per_batch = 1000
```


```python
# %%time
# nr_epochs = 150
# model_1.fit(x_train_xs, y_train_xs, batch_size=samples_per_batch, epochs=nr_epochs,
#             callbacks=[get_tensorboard('Model 1')], verbose=0, validation_data=(x_val, y_val))
```


```python
# %%time
# nr_epochs = 150
# model_2.fit(x_train_xs, y_train_xs, batch_size=samples_per_batch, epochs=nr_epochs,
#             callbacks=[get_tensorboard('Model 2')], verbose=0, validation_data=(x_val, y_val))
```


```python
%%time
nr_epochs = 100
model_1.fit(x_train, y_train, batch_size=samples_per_batch, epochs=nr_epochs,
            callbacks=[get_tensorboard('Model 1 XL')], verbose=0, validation_data=(x_val, y_val))
```


```python
%%time
nr_epochs = 100
model_2.fit(x_train, y_train, batch_size=samples_per_batch, epochs=nr_epochs,
            callbacks=[get_tensorboard('Model 2 XL')], verbose=0, validation_data=(x_val, y_val))
```


```python
%%time
nr_epochs = 100
model_3.fit(x_train, y_train, batch_size=samples_per_batch, epochs=nr_epochs,
            callbacks=[get_tensorboard('Model 2 XL')], verbose=0, validation_data=(x_val, y_val))
```

# Predictions on Individual Images


```python
x_val[0].shape
```


```python
test = np.expand_dims(x_val[0], axis=0)
test.shape
```


```python
np.set_printoptions(precision=3)
```


```python
model_2.predict(test)
```


```python
model_2.predict(x_val).shape
```


```python
model_2.predict_classes(test)
```


```python
y_val[0]
```

#### Write a for loop where you print out the actual value 
#### and the predicted value for the first 10 images in the valuation dataset. 


```python
for number in range(10):
    test_img = np.expand_dims(x_val[number], axis=0)
    predicted_val = model_2.predict_classes(test_img)[0]
    print(f'Actual value: {y_val[number][0]} vs. predicted: {predicted_val}')
```

# Evaluation


```python
model_2.metrics_names
```


```python
test_loss, test_accuracy = model_2.evaluate(x_test, y_test)
print(f'Test loss is {test_loss:0.3} and test accuracy is {test_accuracy:0.1%}')
```

### Confusion Matrix


```python
predictions = model_2.predict_classes(x_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)
```


```python
conf_matrix.shape
```


```python
nr_rows = conf_matrix.shape[0]
nr_cols = conf_matrix.shape[1]
```


```python
conf_matrix.max()
```


```python
conf_matrix.min()
```


```python
conf_matrix[0]
```


```python
plt.figure(figsize=(7,7), dpi=95)
plt.imshow(conf_matrix, cmap=plt.cm.Greens)

plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('Actual Labels', fontsize=12)
plt.xlabel('Predicted Labels', fontsize=12)

tick_marks = np.arange(NR_CLASSES)
plt.yticks(tick_marks, LABEL_NAMES)
plt.xticks(tick_marks, LABEL_NAMES)

plt.colorbar()

for i, j in itertools.product(range(nr_rows), range(nr_cols)):
    plt.text(j, i, conf_matrix[i, j], horizontalalignment='center',
            color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black')
    

plt.show()
```

#### identify the false positives, false negatives, and the true positives in the confusion matrix.


```python
# True Positives
np.diag(conf_matrix)
```


```python
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
recall
```


```python
precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
precision
```


```python
avg_recall = np.mean(recall)
print(f'Model 2 recall score is {avg_recall:.2%}')
```

#### Calculate the average precision for the model as a whole. 
#### Calculate the f-score for the model as a whole. 


```python
avg_precision = np.mean(precision)
print(f'Model 2 precision score is {avg_precision:.2%}')

f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
print(f'Model 2 f score is {f1_score:.2%}')
```


```python

```


```python

```


```python

```


```python

```


```python

```
