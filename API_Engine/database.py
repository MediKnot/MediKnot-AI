"""DATABASE
MongoDB database initialization
"""

# # Installed # #
import mysql.connector

# # Package # #
from .settings import mysql_settings as settings

__all__ = ("client", )


client = mysql.connector.connect(
  host = settings.host,
  user = settings.host,
  password = settings.password,
  database = settings.database
)

