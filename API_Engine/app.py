"""APP
FastAPI app definition, initialization and definition of routes
"""

# # Installed # #
import uvicorn
from subprocess import Popen
from fastapi import FastAPI
from fastapi import status as statuscode
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# # Package # #
from .models import *
from .exceptions import *
from .repositories import DiseaseRepository
from .middlewares import request_handler
from .settings import api_settings as settings

__all__ = ("app", )


app = FastAPI(
    title=settings.title
)
app.middleware("http")(request_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    p = Popen(['pip3', 'install', '-r', 'requirements.txt'])
    p.communicate()

@app.post(
    "/symptomsToDiseases",
    description="Based on symptoms predict top 10 disease",
    tags=["Disease"]
)
def _get_top_10_disease(symptoms: str):
    return DiseaseRepository.getTop10Disease(symptoms)