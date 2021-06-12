import uvicorn
from API_Engine.settings import api_settings as settings

if __name__ == '__main__':
    """Run the API using Uvicorn"""
    # freeze_support()
    uvicorn.run(
        "API_Engine.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload = True
    )