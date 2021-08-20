import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import random as random

BUFFER_SIZE = 70_000
BATCH_SIZE = 128
NUM_EPOCHS = 1

mnist_dataset, mnist_info = tfds.load(
    name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']


def normalize_image(image, label):
    return tf.cast(image, tf.float32) / 255., label


train_and_validation_data = mnist_train.map(normalize_image)
test_data = mnist_test.map(normalize_image)

num_validation_samples = .1 * mnist_info.splits['train'].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


train_and_validation_data = train_and_validation_data.shuffle(BUFFER_SIZE)

train_data = train_and_validation_data.skip(num_validation_samples)
validation_data = train_and_validation_data.take(num_validation_samples)

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)


model = keras.Sequential([
    keras.layers.Conv2D(50, 5, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(50, 3, activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10),
])


model.summary(line_length=75)


loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='auto',
    min_delta=0,
    patience=2,
    verbose=0,
    restore_best_weights=True,
)

model.fit(
    train_data,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping],
    validation_data=validation_data,
    verbose=1,
)


test_lost, test_accuracy = model.evaluate(test_data)

print('Test loss: {0: .4f}. Test accuracy: {1: .2f}%'.format(
    test_lost, test_accuracy*100))


for images, labels in test_data.take(1):
    images_test = images.numpy()
    labels_test = labels.numpy()

    images_plot = np.reshape(images_test, (10000, 28, 28))

    i = random.randint(0, num_test_samples)


plt.figure(figsize=(2, 2))
plt.axis('off')
plt.imshow(images_plot[i-1], cmap="gray", aspect='auto')
plt.show()


print("Label: {}".format(labels_test[i-1]))


predictions = model.predict(images_test[i-1:i])


probabilities = tf.nn.softmax(predictions).numpy()

probabilities = probabilities*100


plt.figure(figsize=(12, 5))
plt.bar(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], height=probabilities[0],
        tick_label=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
plt.show()
