import os
from pathlib import Path
from dotenv import load_dotenv

# Base backend directory
BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent

# Load .env file
ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Application Settings
APP_NAME = os.getenv("APP_NAME", "Chatbot Travel Backend")
APP_ENV = os.getenv("APP_ENV", "development")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

# Database Settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "travel_chatbot_db")

# Data Directories
DATA_DIR = BACKEND_DIR / "data"
INDIAN_PLACES_DIR = DATA_DIR / "indian_places"
INDIAN_TOURISM_DIR = DATA_DIR / "indian_tourism"
TOURIST_SPOTS_DIR = DATA_DIR / "tourist_spots"
