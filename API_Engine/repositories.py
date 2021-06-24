"""REPOSITORIES
Methods to interact with the database
"""
import os

# # Installed # #
from typing import Optional
from fastapi.responses import JSONResponse      
import dialogflow
from google.api_core.exceptions import InvalidArgument
from google.protobuf.json_format import MessageToDict

# # Package # #
from .models import *
from .exceptions import *
from .utils import get_time, get_uuid, diseaseToSymptom, ocr_extraction, aadhar_card_info, prescription_info
from .settings import dialogflow_settings

__all__ = ("DiseaseRepository", "VerificationRepository", "MedicalEventRepository", "DialogFlowRepository")


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
        aadhar_number_extracted = aadhar_card_info(aadhar_card)
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
        extracted_info = prescription_info(prescription)
        return JSONResponse(
            content = extracted_info
        )

class DialogFlowRepository:
    @staticmethod
    def detect_intent_with_parameters(project_id, session_id, query_params, language_code, user_input):
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(project_id, session_id)
        text = user_input
        text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.types.QueryInput(text=text_input)
        response = session_client.detect_intent(session=session, query_input=query_input, query_params=query_params)
        return response

    @staticmethod
    def chat(input_data: str):
        GOOGLE_AUTHENTICATION_FILE_NAME = "../credentials/key.json"
        current_directory = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current_directory, GOOGLE_AUTHENTICATION_FILE_NAME)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path

        GOOGLE_PROJECT_ID = dialogflow_settings.DIALOGFLOW_PROJECT_ID
        session_id = "1234567891"
        context_short_name = "does_not_matter"

        context_name = "projects/" + GOOGLE_PROJECT_ID + "/agent/sessions/" + session_id + "/contexts/" + context_short_name.lower()

        parameters = dialogflow.types.struct_pb2.Struct()

        context_1 = dialogflow.types.context_pb2.Context(
            name=context_name,
            lifespan_count=2,
            parameters=parameters
        )
        query_params_1 = {"contexts": [context_1]}

        language_code = 'en'

        response = DialogFlowRepository.detect_intent_with_parameters(
            project_id=GOOGLE_PROJECT_ID,
            session_id=session_id,
            query_params=query_params_1,
            language_code=language_code,
            user_input=input_data
        )
        result = MessageToDict(response)
        print(result)
        if len(result['queryResult']['fulfillmentMessages']) == 2:
            response = {"message": result['queryResult']['fulfillmentText'],
                        "payload": result['queryResult']['fulfillmentMessages'][1]['payload']}
        else:
            response = {"message": result['queryResult']['fulfillmentText'], "payload": None}
        # response = {"message": result['queryResult']['fulfillmentText'], "payload": None}
        return JSONResponse(
                content = response
            )