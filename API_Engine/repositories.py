"""REPOSITORIES
Methods to interact with the database
"""
# # Installed # #
from typing import Optional
from fastapi.responses import JSONResponse      

# # Package # #
from .models import *
from .exceptions import *
from .utils import get_time, get_uuid, diseaseToSymptom, ocr_extraction, aadhar_card_info

__all__ = ("DiseaseRepository", "VerificationRepository", "MedicalEventRepository")


class DiseaseRepository:
    @staticmethod
    def getTop10Disease(symptom: str):
        diseases = diseaseToSymptom(symptom)
        return JSONResponse(
            content = {
                "diseases": diseases
            }
        )

class VerificationRepository:
    @staticmethod
    def aadharVerification(aadhar_card, aadhar_number: str):
        extracted_info = ocr_extraction(aadhar_card)
        aadhar_number_extracted = aadhar_card_info(extracted_info)
        if aadhar_number_extracted == "Not found!":
            return JSONResponse(
                content = {
                    "status": False,
                    "message": aadhar_number_extracted
                }
            )     
        verification = aadhar_number_extracted == aadhar_number
        return JSONResponse(
            content = {
                "status": verification,
                "message": "Extracted aadhar number: " + aadhar_number_extracted
            }
        )

class MedicalEventRepository:
    @staticmethod
    def prescriptionExtraction(prescription):
        extracted_info = ocr_extraction(prescription).split("string_separation_between_two_extraction")
        return JSONResponse(
            content = {
                "message": extracted_info
            }
        )