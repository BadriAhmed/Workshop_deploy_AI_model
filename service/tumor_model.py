from keras.applications.vgg16 import preprocess_input
from keras.engine.sequential import Sequential
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from loguru import logger
from numpy import ndarray
import sklearn

sklearn.__SKLEARN_SETUP__

from dto.responses import TumorDetectionResponse


def load_keras_model(model_path: str):
    """
    Load a keras model, and return it
    :param model_path:
    :return:
    """
    model = load_model(model_path)
    logger.info("Model Summary ", model.summary())
    return model


def preprocess_image(image_path: str):
    """
    preprocess the image before passing it to the model
    Model input is (224,224,3)
    :param image_path
    :return:
    """
    # load the image
    image = load_img(image_path, target_size=(224, 224, 3))
    logger.info("Image successfully loaded")

    # preprocess the image
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    return image


def make_prediction(model: Sequential, image: ndarray):
    """
    Take a model and an image, and make a prediction as numpy array
    :param model
    :param image
    :return:
    """
    logger.info("Making a Prediction")
    prediction = model.predict(image)
    return prediction


def create_response(prediction: ndarray):
    """
    Format the prediciton to a compatible FastApi Response
    :param prediction: 
    :return: 
    """
    confidence = prediction[0][0]
    prediction = True if prediction > 0.5 else False
    response = TumorDetectionResponse(prediction=prediction,
                                      confidence=confidence)
    return response
