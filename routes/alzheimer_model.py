from fastapi import APIRouter

from service.alzheimer_model import load_model, make_prediction

alzheimer_route = APIRouter()

model = load_model("alzheimer_detection.joblib")


@alzheimer_route.get("/alzheimer_prediction")
def make_alzheimer_prediction(gender: bool, age: int, edu_years: int,
                              ses_status: float, mms_status: float):
    prediction = make_prediction(model, gender, age, edu_years, ses_status, mms_status)

    return prediction
