import tensorflow as tf
from tensorflow.keras import layers, models

# Wczytanie danych
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Przetworzenie danych
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# Budowa modelu CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Ocena modelu
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
