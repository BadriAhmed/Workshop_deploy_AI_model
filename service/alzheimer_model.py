import joblib
import numpy as np

from dto.responses import AlzheimerDetectionResponse


def load_model(model_path: str):
    """
    load scikit learn model
    :param model_path:
    :return:
    """
    model = joblib.load(model_path)
    return model


def make_prediction(model, gender: bool, age: int, edu_years: int,
                    ses_status: float, mms_status: float):
    """
    Use the model to predict whether the patient can have alzaihmer or not
    :param model
    :param gender
    :param age
    :param edu_years: Years of education
    :param ses_status: Socioeconomic Status
    :param mms_status: Mini Mental State Examination
    :return:
    """
    model_input = np.array([[gender, age, edu_years, ses_status, mms_status]])
    prediction = model.predict(model_input)
    return AlzheimerDetectionResponse(prediction=prediction[0])
