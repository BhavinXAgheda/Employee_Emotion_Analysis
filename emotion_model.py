import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Load the FER-2013 dataset
data = pd.read_csv('fer2013.csv')

# Preprocess the data
X = []
y = []
for index, row in data.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='float32')
    pixels = pixels.reshape(48, 48, 1) / 255.0
    X.append(pixels)
    y.append(row['emotion'])

X = np.array(X)
y = to_categorical(np.array(y), num_classes=7)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=64, validation_data=(X_test, y_test))

# Save the model
model.save('emotion_model.h5')
print("Model trained and saved as 'emotion_model.h5'")
