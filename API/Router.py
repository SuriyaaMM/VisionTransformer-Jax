from fastapi import APIRouter
from API.Schema import TrainResponse
from VIT.Train import Train

Router = APIRouter()

@Router.get("/")
def Home():
    return {"message" : "VisionTransformer-Jax API is running"}

@Router.get("/train", response_model = TrainResponse)
def RunTrain():
    vit, params, accuracy = Train()
    return {"message" : "VisionTransformrer training Completed Successfully",
            "accuracy" : accuracy}


