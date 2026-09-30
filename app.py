import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import APP_NAME, APP_ENV, HOST, PORT
from database.db import get_db, init_engine
from database.models import (
    Base, ChatSession, UserQuery, UserPreference, 
    Itinerary, ItineraryItem, UserFeedback
)
from services.nlp_service import nlp_extractor
from services.recommendation_service import destination_recommender
from services.poi_service import poi_service

app = FastAPI(
    title=APP_NAME,
    description="Offline ML-Based Personalized Travel Planning Chatbot API",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Request & Response Schemas
class ChatRequest(BaseModel):
    message: str = Field(..., description="User prompt text", min_length=1)
    session_id: Optional[str] = Field(default=None, description="Optional conversation session ID")
    active_interests: Optional[List[str]] = Field(default=None, description="Persisted user interests if previously extracted")


class ExtractedPreferencesSchema(BaseModel):
    days: Optional[int] = None
    budget: Optional[float] = None
    interests: List[str] = []
    season: Optional[str] = None
    group_type: Optional[str] = None
    destination: Optional[str] = None
    is_modification: Optional[bool] = False
    remove_categories: Optional[List[str]] = []
    add_interests: Optional[List[str]] = []


class DestinationRecommendationSchema(BaseModel):
    id: int
    rank: int
    badge: str
    destination_name: str
    state: str
    region: Optional[str] = None
    match_percentage: int
    trip_types: List[str] = []
    best_seasons: List[str] = []
    ideal_days: int
    daily_budget_range: List[int] = []
    primary_attractions: List[str] = []
    activities_available: List[str] = []
    unique_experiences: Optional[str] = None


class POISchema(BaseModel):
    id: int
    name: str
    destination: str
    type: str
    time_needed_hrs: float
    rating: float
    entrance_fee_inr: float
    significance: str
    best_time_to_visit: str
    dslr_allowed: Optional[str] = "Yes"
    characteristics: List[str] = []
    description: Optional[str] = None
    match_score: int = 75
    match_level: str = "GOOD MATCH"
    is_high_match: bool = False
    matched_interests: List[str] = []
    match_badge: str = "⭐ GOOD MATCH"
    match_explanation: Optional[str] = None
    why_poi: Optional[List[str]] = []


class DayPlanSchema(BaseModel):
    day: int
    pois: List[POISchema]
    total_time_hrs: float
    total_cost: float


class CustomItinerarySchema(BaseModel):
    destination: str
    days: int
    budget: float
    total_cost: float
    within_budget: bool
    max_hours_per_day: float = 8.0
    day_plans: List[DayPlanSchema]
    why_destination: List[str] = []


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    extracted_preferences: ExtractedPreferencesSchema
    selected_destination: Optional[str] = None
    destinations: List[DestinationRecommendationSchema] = []
    pois: List[POISchema] = []
    itinerary: Optional[CustomItinerarySchema] = None
    query_id: int
    db_saved: bool


class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    feedback_text: Optional[str] = None
    itinerary_id: Optional[int] = None


def format_day_wise_itinerary(itinerary_data: Dict[str, Any], modification_note: Optional[str] = None) -> str:
    """
    Formats natural day-wise text output for the chatbot display matching specification formatting.
    Ensures place, time slots, and daily budget breakdown for all requested days.
    """
    dest = itinerary_data["destination"].upper()
    days = itinerary_data["days"]
    budget = itinerary_data["budget"]
    total_cost = itinerary_data["total_cost"]
    daily_budget_est = round(budget / days, 2)

    reply = ""
    if modification_note:
        reply += f"🔄 **ITINERARY MODIFICATION DETECTED**\n*{modification_note}*\n\n"

    reply += f"✈️ **YOUR {days}-DAY {dest} TRIP ITINERARY**\n"
    reply += f"💰 **Total Budget:** ₹{budget:,.0f} | **Est. Sightseeing Fees:** ₹{total_cost:,.0f} | **Daily Budget Target:** ₹{daily_budget_est:,.0f}/day\n"
    reply += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    emoji_map = {
        "Valley": "🏔️", "Mountain": "❄️", "Pass": "🏔️", "Temple": "🛕", 
        "Waterfall": "🌊", "Market": "🛍️", "Trek": "🥾", "Nature": "🌿",
        "Beach": "🏖️", "Fort": "🏰", "Museum": "🏛️", "Hot Springs": "♨️",
        "Trek Point": "🥾", "Sightseeing Point": "📍", "Promenade": "🌅", "Heritage Walk": "🏛️"
    }

    for day_plan in itinerary_data.get("day_plans", []):
        pois = day_plan.get("pois", [])
        d_num = day_plan["day"]
        day_fee_sum = day_plan.get("total_cost", 0.0)
        
        reply += f"📅 **DAY {d_num}** (Sightseeing: {day_plan.get('total_time_hrs', 0.0)} Hours | Day Fees: ₹{day_fee_sum:,.0f})\n\n"

        if not pois:
            reply += "  📍 Local exploration & scenic relaxation in city center\n\n"
            reply += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            continue

        for p in pois:
            p_type = p.get("type", "Attraction")
            icon = emoji_map.get(p_type, "📍")
            badge = " ⭐ Recommended" if p.get("is_high_match") else ""
            fee_val = p.get("entrance_fee_inr", 0.0)
            fee_str = "Free Entrance" if fee_val == 0 else f"Entrance Fee: ₹{fee_val:,.0f}"
            time_slot = p.get("time_slot") or f"⏰ {p.get('best_time_to_visit', 'Daytime')} ({p.get('time_needed_hrs', 2)} Hrs)"

            reply += f"  {icon} **{p['name']}**{badge}\n"
            reply += f"     🕒 Time: {time_slot} | ⭐ Rating: {p.get('rating', 4.5)}/5.0 | 💰 {fee_str}\n"
            if p.get("description"):
                reply += f"     ℹ️ {p['description']}\n"
            reply += "\n"

        reply += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    # Explainable AI Reasoning
    reply += "💡 **EXPLAINABLE AI REASONING**\n"
    reply += f"**Why {itinerary_data['destination']} for {days} Days?**\n"
    for why in itinerary_data.get("why_destination", []):
        reply += f"{why}\n"

    all_pois = itinerary_data.get("ranked_pois", [])
    if all_pois:
        top_poi = all_pois[0]
        reply += f"\n**Highlight Spot: Why {top_poi['name']}?**\n"
        for why_p in top_poi.get("why_poi", []):
            reply += f"{why_p}\n"

    return reply


@app.get("/")
def read_root():
    return {
        "status": "online",
        "app": APP_NAME,
        "environment": APP_ENV,
        "message": "Welcome to the Offline Travel Chatbot API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "travel_chatbot_backend"
    }


@app.post("/api/chat/message", response_model=ChatResponse)
def handle_chat_message(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Main Chat Message Endpoint:
    1️⃣ Destination Selection / Detection
    7️⃣ Find POIs (indian_places.csv)
    8️⃣ Add POI Characteristics (tourist_spots.csv) & Match Interests
    9️⃣ POI Recommendation Scoring (Max 100)
    🔟 Custom Itinerary Algorithm (Max 8 hrs/day)
    1️⃣1️⃣ Budget Checking (<= Budget)
    1️⃣2️⃣ Final Day-Wise Itinerary
    1️⃣3️⃣ Explainable AI
    1️⃣4️⃣ User Modifies Itinerary (e.g. Remove temples and add adventure)
    🗄️ MySQL Persistence (chat_sessions, user_preferences, itineraries, itinerary_items)
    """
    session_id = req.session_id or str(uuid.uuid4())

    # Ensure ChatSession exists in DB
    try:
        session_obj = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session_obj:
            session_obj = ChatSession(session_id=session_id)
            db.add(session_obj)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DATABASE] ChatSession init note: {e}")

    # 1. NLP Extraction
    extracted = nlp_extractor.extract_preferences(req.message)

    # Load prior preferences from session if available
    prior_pref = None
    try:
        prior_pref = db.query(UserPreference).filter(UserPreference.session_id == session_id).order_by(UserPreference.created_at.desc()).first()
    except Exception:
        pass

    # Merge active interests & destination from context
    active_interests = list(extracted.get("interests", []))
    if not active_interests and req.active_interests:
        active_interests = req.active_interests
    elif prior_pref and not active_interests:
        active_interests = prior_pref.interests or []

    # Handle 1️⃣4️⃣ User Modification Intent (e.g. "Remove temples and add more adventure")
    removed_categories = list(extracted.get("remove_categories", []))
    add_interests = list(extracted.get("add_interests", []))

    if add_interests:
        for ai in add_interests:
            if ai not in active_interests:
                active_interests.append(ai)

    if removed_categories:
        active_interests = [i for i in active_interests if i.lower() not in removed_categories]

    extracted["interests"] = active_interests

    # Determine Destination
    selected_dest = extracted.get("destination")
    if not selected_dest and prior_pref and prior_pref.selected_destination:
        selected_dest = prior_pref.selected_destination

    days = extracted.get("days") or (prior_pref.days if prior_pref and prior_pref.days else None) or 4
    
    extracted_budget = extracted.get("budget")
    if extracted_budget is not None and extracted_budget >= 100.0:
        budget = extracted_budget
    elif prior_pref and prior_pref.budget and prior_pref.budget >= 2000.0:
        budget = prior_pref.budget
    else:
        budget = max(20000.0, days * 3000.0)

    season = extracted.get("season") or (prior_pref.season if prior_pref else None)

    recommended_destinations = []
    pois = []
    custom_itinerary = None
    reply_text = ""

    # 2. Flow Decision: Destination Selected vs ML Destination Search
    if not selected_dest:
        recommended_destinations = destination_recommender.recommend(extracted, top_k=3)
        if recommended_destinations:
            selected_dest = recommended_destinations[0]["destination_name"]

    if selected_dest:
        # Generate 1️⃣0️⃣ Custom Itinerary with 8h/day limit & 1️⃣1️⃣ Budget Check & 1️⃣4️⃣ Modifications
        custom_itinerary = poi_service.generate_custom_itinerary(
            destination_name=selected_dest,
            days=days,
            budget=budget,
            user_interests=active_interests,
            removed_categories=removed_categories,
            user_season=season
        )
        pois = custom_itinerary.get("ranked_pois", [])

        modification_note = None
        if extracted.get("is_modification"):
            mod_parts = []
            if removed_categories:
                mod_parts.append(f"Removed: {', '.join(removed_categories).capitalize()}")
            if add_interests:
                mod_parts.append(f"Added: {', '.join(add_interests).capitalize()}")
            modification_note = " | ".join(mod_parts)

        reply_text = format_day_wise_itinerary(custom_itinerary, modification_note)

    # 3. 🗄️ Database Persistence
    db_saved = False
    query_id = 0
    try:
        # Save User Query
        user_query = UserQuery(session_id=session_id, raw_text=req.message)
        db.add(user_query)
        db.flush()

        # Save User Preference
        pref_obj = UserPreference(
            session_id=session_id,
            query_id=user_query.id,
            selected_destination=selected_dest,
            days=days,
            budget=budget,
            interests=active_interests,
            removed_categories=removed_categories,
            season=season,
            group_type=extracted.get("group_type")
        )
        db.add(pref_obj)

        # Save Itinerary & Itinerary Items if generated
        if custom_itinerary and selected_dest:
            itin_obj = Itinerary(
                session_id=session_id,
                destination=selected_dest,
                days=days,
                total_budget=budget,
                total_cost=custom_itinerary.get("total_cost", 0.0)
            )
            db.add(itin_obj)
            db.flush()

            for day_plan in custom_itinerary.get("day_plans", []):
                for p in day_plan.get("pois", []):
                    item_obj = ItineraryItem(
                        itinerary_id=itin_obj.id,
                        day_number=day_plan["day"],
                        poi_name=p["name"],
                        poi_type=p.get("type"),
                        time_needed_hrs=p.get("time_needed_hrs", 2.0),
                        entrance_fee=p.get("entrance_fee_inr", 0.0),
                        score=p.get("total_score", 75.0),
                        match_explanation=p.get("match_explanation")
                    )
                    db.add(item_obj)

        db.commit()
        query_id = user_query.id
        db_saved = True
    except Exception as e:
        db.rollback()
        print(f"[DATABASE] Error persisting chat session: {e}")

    return ChatResponse(
        session_id=session_id,
        reply=reply_text,
        extracted_preferences=ExtractedPreferencesSchema(
            days=days,
            budget=budget,
            interests=active_interests,
            season=season,
            group_type=extracted.get("group_type"),
            destination=selected_dest,
            is_modification=extracted.get("is_modification"),
            remove_categories=removed_categories,
            add_interests=add_interests
        ),
        selected_destination=selected_dest,
        destinations=[DestinationRecommendationSchema(**d) for d in recommended_destinations],
        pois=[POISchema(**p) for p in pois],
        itinerary=CustomItinerarySchema(**custom_itinerary) if custom_itinerary else None,
        query_id=query_id,
        db_saved=db_saved
    )


@app.post("/api/feedback")
def submit_user_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Saves user rating and feedback to database user_feedback table."""
    try:
        fb = UserFeedback(
            session_id=req.session_id,
            itinerary_id=req.itinerary_id,
            rating=req.rating,
            feedback_text=req.feedback_text
        )
        db.add(fb)
        db.commit()
        return {"status": "success", "message": "Thank you for your feedback!"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pois/destination", response_model=List[POISchema])
def get_destination_pois(req: Dict[str, Any]):
    """Direct endpoint to retrieve POIs and interest matches for any destination."""
    destination = req.get("destination", "Manali")
    interests = req.get("interests", [])
    limit = req.get("limit", 15)
    pois = poi_service.get_pois_for_destination(
        destination_name=destination,
        user_interests=interests,
        limit=limit
    )
    return [POISchema(**p) for p in pois]


@app.get("/api/preferences/recent")
def get_recent_preferences(limit: int = 10, db: Session = Depends(get_db)):
    """Fetches recently stored user queries and preferences from the database."""
    queries = db.query(UserQuery).order_by(UserQuery.created_at.desc()).limit(limit).all()
    return [q.to_dict() for q in queries]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)


