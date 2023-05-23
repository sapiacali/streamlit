import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the shape classifier model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # Convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
model.add(Conv2D(64, (3, 3), activation='relu'))  # Convolutional layer
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling layer
model.add(Flatten())  # Flatten layer
model.add(Dense(64, activation='relu'))  # Fully connected layer
model.add(Dense(3, activation='softmax'))  # Output layer with 3 classes: circle, square, triangle

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the dataset using ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)  # Rescale the pixel values between 0 and 1

train_generator = train_datagen.flow_from_directory(
    'shapes/train',  # Path to the directory containing the training images
    target_size=(64, 64),  # Resize the images to 64x64
    batch_size=32,
    class_mode='categorical'  # Use categorical labels
)

# Train the model
model.fit(train_generator, epochs=10)

# Save the model in HDF5 format
model.save('shape_classifier.h5')