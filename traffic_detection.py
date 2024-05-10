 import pickle
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
import os

def network():
    """
    Create the network.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(4))  # Updated to output 4 classes
    model.add(Activation('softmax'))  # Use softmax activation for multi-class classification
    return model


def train(file_path, model):
    """
    Train the network.
    """
    x_, y_ = pickle.load(open(file_path, "rb"))
    random_state = 130
    X_train, x_validation, y_train, y_validation = train_test_split(x_, y_, train_size=0.80,
                                                                    test_size=0.2,
                                                                    random_state=random_state)
    # preprocess data
    X_train_normalized = np.array(X_train / 255.0 - 0.5)
    label_binarizer = LabelBinarizer()
    y_train_one_hot = label_binarizer.fit_transform(y_train)

    model.summary()
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_normalized, y_train_one_hot, epochs=20, validation_split=0.2)

    model.save('/content/model (1).h5')
    return history

def test(file_path, model):
    try:
        X_test, y_test = pickle.load(open(file_path, "rb"))

        # preprocess data
        X_normalized_test = np.array(X_test / 255.0 - 0.5)
        label_binarizer = LabelBinarizer()
        y_one_hot_test = label_binarizer.fit_transform(y_test)

        print("Testing")

        metrics = model.evaluate(X_normalized_test, y_one_hot_test)
        for metric_i in range(len(model.metrics_names)):
            metric_name = model.metrics_names[metric_i]
            metric_value = metrics[metric_i]
            print('{}: {}'.format(metric_name, metric_value))
    except Exception as e:
        print("Error loading test data:", e)


def test_an_image(file_path, model):
    """
    Resize the input image to [32, 32, 3], then feed it into the NN for prediction.
    :param file_path: Path to the image file.
    :param model: Trained Keras model.
    :return: Predicted state.
    """
    desired_dim = (32, 32)
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
    img_ = np.expand_dims(np.array(img_resized), axis=0)

    # Predict probabilities for each class
    predicted_probabilities = model.predict(img_)

    # Extract the class with the highest probability
    predicted_class_index = np.argmax(predicted_probabilities)

    return predicted_class_index

def send_stop_signal():
    print("Stop signal sent.")

# Function to send go signal
def send_go_signal():
    print("Go signal sent.")


if __name__ == "__main__":
    train_file = "/content/bosch_udacity_train.p"
    test_file = "/content/bosch_udacity_train.p"
    model_file = "/content/model (1).h5"

    if os.path.exists(model_file):
        model = load_model(model_file)
        print("Model loaded successfully.")
    else:
        model = network()
        train(train_file, model)
        print("Model trained successfully and saved.")

    # Test the network
    test(test_file, model=model)

    # Test with a single image
    demo_flag = True
    file_path = '/content/green.jpg'
    states = ['red', 'yellow', 'green', 'off']
    if demo_flag:
        predicted_state = test_an_image(file_path, model=model)
        if predicted_state == 0:  # Red detected
            print("Red")
            send_stop_signal()
        elif predicted_state == 2:  # Green detected
            print("Green")
            send_go_signal()
