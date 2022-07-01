from fastapi import FastAPI
from uvicorn import Server, Config

from routes.alzheimer_model import alzheimer_route
from routes.tumor_predictions import tumor_route

app = FastAPI()
app.include_router(tumor_route)
app.include_router(alzheimer_route)


if __name__ == '__main__':
    # Setting up uvicorn server
    server = Server(Config(app=app, host="0.0.0.0", port=8000))
    # Overriding uvicorn logging system
    server.run()
