from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    preferences = relationship("UserPreference", back_populates="session", cascade="all, delete-orphan")
    itineraries = relationship("Itinerary", back_populates="session", cascade="all, delete-orphan")
    feedback = relationship("UserFeedback", back_populates="session", cascade="all, delete-orphan")


class UserQuery(Base):
    __tablename__ = "user_queries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(100), index=True, nullable=True)
    raw_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to extracted preferences
    preferences = relationship("UserPreference", back_populates="query", cascade="all, delete-orphan", uselist=False)

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "raw_text": self.raw_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "preferences": self.preferences.to_dict() if self.preferences else None
        }


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.session_id"), nullable=True)
    query_id = Column(Integer, ForeignKey("user_queries.id"), nullable=True)
    selected_destination = Column(String(100), nullable=True)
    days = Column(Integer, nullable=True)
    budget = Column(Float, nullable=True)
    interests = Column(JSON, nullable=True)  # List of strings e.g. ["nature", "adventure"]
    removed_categories = Column(JSON, nullable=True) # List of strings e.g. ["temple"]
    season = Column(String(50), nullable=True)
    group_type = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("ChatSession", back_populates="preferences")
    query = relationship("UserQuery", back_populates="preferences")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "query_id": self.query_id,
            "selected_destination": self.selected_destination,
            "days": self.days,
            "budget": self.budget,
            "interests": self.interests or [],
            "removed_categories": self.removed_categories or [],
            "season": self.season,
            "group_type": self.group_type,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Itinerary(Base):
    __tablename__ = "itineraries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.session_id"), nullable=False)
    destination = Column(String(100), nullable=False)
    days = Column(Integer, nullable=False, default=4)
    total_budget = Column(Float, nullable=False, default=20000.0)
    total_cost = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("ChatSession", back_populates="itineraries")
    items = relationship("ItineraryItem", back_populates="itinerary", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "destination": self.destination,
            "days": self.days,
            "total_budget": self.total_budget,
            "total_cost": self.total_cost,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "items": [item.to_dict() for item in self.items]
        }


class ItineraryItem(Base):
    __tablename__ = "itinerary_items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    itinerary_id = Column(Integer, ForeignKey("itineraries.id"), nullable=False)
    day_number = Column(Integer, nullable=False)
    poi_name = Column(String(150), nullable=False)
    poi_type = Column(String(100), nullable=True)
    time_needed_hrs = Column(Float, nullable=False, default=2.0)
    entrance_fee = Column(Float, nullable=False, default=0.0)
    score = Column(Float, nullable=False, default=75.0)
    match_explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    itinerary = relationship("Itinerary", back_populates="items")

    def to_dict(self):
        return {
            "id": self.id,
            "itinerary_id": self.itinerary_id,
            "day_number": self.day_number,
            "poi_name": self.poi_name,
            "poi_type": self.poi_type,
            "time_needed_hrs": self.time_needed_hrs,
            "entrance_fee": self.entrance_fee,
            "score": self.score,
            "match_explanation": self.match_explanation,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("chat_sessions.session_id"), nullable=False)
    itinerary_id = Column(Integer, ForeignKey("itineraries.id"), nullable=True)
    rating = Column(Integer, nullable=False) # 1-5 stars
    feedback_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("ChatSession", back_populates="feedback")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "itinerary_id": self.itinerary_id,
            "rating": self.rating,
            "feedback_text": self.feedback_text,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


