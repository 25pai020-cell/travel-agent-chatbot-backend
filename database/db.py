import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, BACKEND_DIR
from database.models import Base

logger = logging.getLogger("travel_chatbot.database")

# Construct MySQL Database URL
if DB_PASSWORD:
    MYSQL_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    MYSQL_SERVER_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"
else:
    MYSQL_DATABASE_URL = f"mysql+pymysql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    MYSQL_SERVER_URL = f"mysql+pymysql://{DB_USER}@{DB_HOST}:{DB_PORT}"

# Local SQLite fallback path if MySQL is unavailable
SQLITE_PATH = BACKEND_DIR / "travel_chatbot.db"
SQLITE_DATABASE_URL = f"sqlite:///{SQLITE_PATH}"


def init_engine():
    """
    Attempts to initialize MySQL engine.
    If MySQL server is reachable, ensures DB exists.
    Falls back gracefully to SQLite if MySQL is unavailable.
    """
    engine = None
    try:
        # Step 1: Ensure MySQL Database exists
        server_engine = create_engine(MYSQL_SERVER_URL, isolation_level="AUTOCOMMIT")
        with server_engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
        server_engine.dispose()

        # Step 2: Connect to the specific database
        engine = create_engine(MYSQL_DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
        logger.info(f"Connected to MySQL database: {DB_NAME} on {DB_HOST}:{DB_PORT}")
        print(f"[DATABASE] Connected to MySQL database: {DB_NAME}")
    except Exception as e:
        logger.warning(f"MySQL connection not ready ({e}). Falling back to local SQLite database.")
        print(f"[DATABASE] MySQL not reachable ({e}). Using local SQLite database at: {SQLITE_PATH}")
        engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})

    # Create tables
    Base.metadata.create_all(bind=engine)
    return engine


engine = init_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI Dependency for database session management"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
