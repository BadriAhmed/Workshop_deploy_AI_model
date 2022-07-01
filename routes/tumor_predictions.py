from fastapi import APIRouter, File

from service.tumor_model import load_keras_model, preprocess_image, make_prediction, create_response

tumor_route = APIRouter()

model = load_keras_model("tumor_detection_model.h5")


@tumor_route.post("/tumor_prediction")
def make_tumor_prediction(file: bytes = File(...)):
    with open('image.jpg', 'wb') as image:
        image.write(file)
        image.close()

    image = preprocess_image("image.jpg")

    prediction = make_prediction(model, image)

    response = create_response(prediction)

    return response
