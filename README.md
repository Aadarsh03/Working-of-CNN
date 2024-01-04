# Convolutional Neural Network (CNN) for MNIST Classification

This repository contains the code for implementing a Convolutional Neural Network (CNN) for classifying handwritten digits in the MNIST dataset using TensorFlow and Keras.

## Steps to Reproduce:

### 1. Load and Preprocess the MNIST Dataset:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images=train_images.reshape((60000, 28, 28,1))
test_images=test_images.reshape((10000, 28, 28,1))
train_images=train_images.astype('float32')/255
test_images=test_images.astype('float32')/255
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
```

### 2. Create a Simple CNN Model:

```python
model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation ='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers. Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

- Define a CNN model using TensorFlow and Keras.
- Add Convolutional and Pooling layers.
- Flatten the output and add Dense layers for classification.

### 3. Compile and Train the Model:

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))
```
- Compile the model with the Adam optimizer and categorical crossentropy loss.
- Train the model on the training data for a specified number of epochs.

### 4. Evaluate the Model:

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc * 100:.2f}%')
```
- Evaluate the trained model on the test data.
- Print the test accuracy.

## 6. Make Predictions:

```python
predictions = model.predict(test_images[1:25])

print('Prediction\n', predictions)
print('\nThresholded output\n', (predictions > 0.5).astype(int))
```
- Use the trained model to make predictions on the test images.


## Files:

- `P2_CNN_Working_MnistDataset.ipynb`: Jupyter notebook containing the entire code.
- `README.md`: This README file.

## Instructions for Use:

1. Open the Jupyter notebook `P2_CNN_Working_MnistDataset.ipynb`.
2. Follow the code cells sequentially for data preparation, model building, evaluation, and predictions.
3. Modify parameters or experiment as needed.
4. Save the notebook if you make changes.

## Note:

- Ensure you have the necessary Python libraries installed.
- Modify file paths and names accordingly.

