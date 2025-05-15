from fastapi import FastAPI
from API.Router import Router
import uvicorn

VITAPI = FastAPI()
VITAPI.include_router(Router)

if __name__ == "__main__":
    uvicorn.run("API.Main:VITAPI", reload = True)


