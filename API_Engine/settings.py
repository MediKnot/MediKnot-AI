"""SETTINGS
Settings loaders using Pydantic BaseSettings classes (load from environment variables / dotenv file)
"""

# # Installed # #
import pydantic

__all__ = ("api_settings", "mysql_settings", "dialogflow_settings")


class BaseSettings(pydantic.BaseSettings):
    class Config:
        env_file = ".env"


class APISettings(BaseSettings):
    title: str = "MediKnot AI API"
    host: str = "0.0.0.0"
    port: int = 5003
    log_level: str = "INFO"

    class Config(BaseSettings.Config):
        env_prefix = "API_"


class MySQLSettings(BaseSettings):
    host: str = "20.198.81.29"
    user: str = "dev"
    password: str = "tvs@mitra"
    database: str = "tvs-mitra"
    
    class Config(BaseSettings.Config):
        env_prefix = "MYSQL_"

class DialogFlowSettings(BaseSettings):
    DIALOGFLOW_PROJECT_ID : str = "docbot-kopk"
    
    class Config(BaseSettings.Config):
        env_prefix = "DIALOGFLOW_"


api_settings = APISettings()
mysql_settings = MySQLSettings()
dialogflow_settings = DialogFlowSettings()