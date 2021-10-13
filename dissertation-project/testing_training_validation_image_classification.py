#  Use convolutional neural network for training clasify of image. it train the image on the dataset which inside the data folder.
#  there have been used 3 types of data for testing,training,validation those will classify the data and provide the information of accuracy data.

import tensorflow as tf  #TensorFlow allows developers to create dataflow graphs—structures that describe how data moves through a graph, or a series of processing nodes.
from tensorflow import keras #The core data structures of Keras are layers and models.
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import numpy as np  # NumPy is the fundamental package for scientific computing in Python.It can be used to perform a wide variety of mathematical operations on arrays.

import matplotlib.pyplot as plt # matplotlib.pyplot is a collection of functions that make matplotlib work like MATLAB.

# training the data by swquential model and override the training step function of the Model class by fit() method.
def training_network(img_width, img_height):
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = \
        val_datagen.flow_from_directory('data/products_validation'
                                        , target_size=(img_width, img_height), batch_size=32, class_mode='categorical')

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = \
        train_datagen.flow_from_directory('data/products_training'
                                          , target_size=(img_width, img_height), batch_size=32,
                                          class_mode='categorical')

    # A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
    model = Sequential()
    print(model)

    # Convolution - extracting appropriate features from the input image.
    model.add(Conv2D(16, (3, 3), input_shape=(img_width, img_height, 3), activation='relu', padding='valid'))
    # Pooling: reduces dimensionality of the feature maps but keeps the most important information.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 3, activation='relu')),
    model.add(MaxPooling2D()),
    model.add(Conv2D(32, 3, activation='relu')),
    model.add(MaxPooling2D()),
    model.add(Flatten()),
    model.add(Dense(128, activation='relu')),
    # Flattening layer to arrange 3D volumes into a 1D vector.
    model.add(Flatten())
    # Fully connected layers: ensures connections to all activations in the previous layer.
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_generator,epochs=100,
                             validation_data=val_generator,
                             validation_steps=20)

    # list all data in history
    print(history.history.keys())
    plt.plot(history.history['loss'][:])
    plt.plot(history.history['val_loss'][:])
    plt.legend(['train', 'val'], loc='upper left')

    plt.ylabel('MSE loss')
    plt.xlabel('epoch')
    plt.title('Loss function MSE')

    image_path = "training_output.png"
    plt.savefig(image_path)
    plt.clf()

    model.save("image_classification.h5")
    return model


def load_saved_model(model_name):
    model = load_model(model_name)
    print(model.summary())
    return model


def test_trained_network(model,img_width, img_height):
    # get the testing accuracy data
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(directory='data/products_testing',
                                                      target_size=(img_width, img_height),
                                                      color_mode="rgb", batch_size=32, class_mode="categorical",
                                                      shuffle=False)
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    scores_test = model.evaluate(test_generator,
                                                steps=STEP_SIZE_TEST)
    print("Testing accuracy = ", scores_test[1])

    # Get the training accuracy data
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(directory='data/products_training',
                                                    target_size=(img_width, img_height),
                                                    color_mode="rgb", batch_size=32, class_mode="categorical",
                                                    shuffle=False)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    scores_train = model.evaluate(train_generator, steps=STEP_SIZE_TRAIN)
    print("Training accuracy = ", scores_train[1])

    # Get The Validation accuracy data
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(directory='data/products_validation',target_size=(img_width, img_height),
                                                      color_mode="rgb", batch_size=32, class_mode="categorical",shuffle=False)

    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size
    scores_val = model.evaluate(val_generator,steps=STEP_SIZE_VAL)
    print("Validation accuracy = ", scores_val[1])




def predict_class_file(model, img_str, img_width, img_height):
    # PREDICT THE CLASS OF ONE IMAGE
    img = image.load_img(img_str, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes_predic = model.predict(images)
    classes_predic = np.argmax(classes_predic,axis=1)
    class_names = ("bottle", "fruit",)
    print("Predicted product = ",classes_predic[0], class_names[classes_predic[0]])
    return classes_predic[0], class_names[classes_predic[0]]


def main():
    product_img_width = 64
    product_img_height = 64
    # print the summary of classify dataset
    model = load_saved_model("image_classification.h5")
    # train the dataset by model and layers. it will save the classification of data in file.
    model = training_network(product_img_width, product_img_height)
    # get the accuracy of data testing,training,validation.
    test_trained_network(model, product_img_width, product_img_height)
    # Name of the classes.
    class_names = ("bottle", "fruit",)
    # show the predication of classify data image.
    class_code, class_name = predict_class_file(model, "current_image.png", product_img_width, product_img_height)


if __name__ == '__main__':
    main()  