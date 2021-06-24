"""APP
FastAPI app definition, initialization and definition of routes
"""

# # Installed # #
import uvicorn
from subprocess import Popen
from fastapi import FastAPI, File, UploadFile
from fastapi import status as statuscode
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# # Package # #
from .models import *
from .exceptions import *
from .repositories import DiseaseRepository, VerificationRepository, MedicalEventRepository, DialogFlowRepository
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

@app.post(
    "/aadharVerification",
    status_code=statuscode.HTTP_201_CREATED,
    tags=["Files"]
    )
async def _aadhar_verification(aadhar_number: str, aadhar_card: UploadFile = File(...)):
    return VerificationRepository.aadharVerification(aadhar_card, aadhar_number)

@app.post(
    "/prescription",
    status_code=statuscode.HTTP_201_CREATED,
    tags=["Files"]
    )
async def _prescription_extraction(prescription: UploadFile = File(...)):
    return MedicalEventRepository.prescriptionExtraction(prescription)


@app.post(
    "/chat",
    tags=["DialogFlow"]
    )
async def _get_response_for_query(text: str):
    return DialogFlowRepository.chat(text)