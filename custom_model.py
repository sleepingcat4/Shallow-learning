from tensorflow import keras
from tensorflow.keras import layers

num_classes=10

# Define a custom deep learning architecture
model_custom = keras.Sequential(
    [
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(units=512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(units=num_classes, activation="softmax"),
    ]
)

# Compile the custom model
model_custom.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model_custom.summary()

# Train the custom model
history_custom = model_custom.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

