"""REPOSITORIES
Methods to interact with the database
"""
# # Installed # #
from typing import Optional
from fastapi.responses import JSONResponse      

# # Package # #
from .models import *
from .exceptions import *
from .utils import get_time, get_uuid, diseaseToSymptom

__all__ = ("DiseaseRepository",)


class DiseaseRepository:
    @staticmethod
    def getTop10Disease(symptom: str):
        diseases = diseaseToSymptom(symptom)
        return JSONResponse(
            content = {
                "diseases": diseases
            }
        )