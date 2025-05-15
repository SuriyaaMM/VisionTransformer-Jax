from pydantic import BaseModel

class TrainResponse(BaseModel):
    message: str
    accuracy: float


