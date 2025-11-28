from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# --- DATABASE CONFIGURATION ---
# Switched to the synchronous psycopg2 driver
DATABASE_URL = "postgresql://suryap:suryap@localhost/hecker"

# Create a synchronous SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=True)

# Create a configured "Session" class for creating DB sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a base class for our declarative models
Base = declarative_base()

# Dependency to get a DB session in API endpoints
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
