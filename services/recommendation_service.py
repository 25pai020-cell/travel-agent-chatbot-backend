import json
import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import INDIAN_TOURISM_DIR, DATA_DIR


class DestinationRecommender:
    """
    Offline ML-Based Destination Recommendation Service.
    Uses TF-IDF feature extraction, Cosine Similarity matching, and multi-factor
    scoring (season compatibility, duration penalty, and budget feasibility).
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        if dataset_path is None:
            self.dataset_path = INDIAN_TOURISM_DIR / "india_tourism_dataset.json"
        else:
            self.dataset_path = Path(dataset_path)

        self.destinations: List[Dict[str, Any]] = []
        self.corpus: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        self._load_and_index_dataset()

    def _load_and_index_dataset(self):
        """Loads JSON dataset and builds the TF-IDF feature index."""
        if not self.dataset_path.exists():
            print(f"[RECOMMENDER] Warning: Dataset not found at {self.dataset_path}")
            return

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.destinations = json.load(f)

        print(f"[RECOMMENDER] Loaded {len(self.destinations)} destinations for ML indexing.")

        # Build composite feature text for each destination
        self.corpus = []
        for d in self.destinations:
            # Collect text fields
            name = d.get("destination_name", "")
            state = d.get("state", "")
            region = d.get("region", "")
            trip_types = " ".join(d.get("trip_types", []))
            activities = " ".join(d.get("activities_available", []))
            attractions = " ".join(d.get("primary_attractions", []))
            hidden_gems = " ".join(d.get("hidden_gems", []))
            experiences = d.get("unique_experiences", "")
            seasons = " ".join(d.get("best_seasons", []))
            ideal_for = " ".join(d.get("ideal_for", []))

            # Normalize underscores (e.g. Hill_station -> hill station)
            composite_text = f"{name} {state} {region} {trip_types} {activities} {attractions} {hidden_gems} {experiences} {seasons} {ideal_for}"
            composite_text = composite_text.replace("_", " ").lower()

            self.corpus.append(composite_text)

        # Fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        print("[RECOMMENDER] TF-IDF indexing successfully built.")

    def _compute_season_score(self, destination: Dict[str, Any], target_season: Optional[str]) -> float:
        """Evaluates season suitability (0.0 to 1.0)."""
        if not target_season:
            return 1.0

        target_season = target_season.lower()
        best_seasons = [s.lower() for s in destination.get("best_seasons", [])]
        avoid_seasons = [s.lower() for s in destination.get("avoid_seasons", [])]
        peak = destination.get("peak_tourist_season", "").lower()

        # Check direct match in best seasons
        if any(target_season in s for s in best_seasons):
            return 1.0
        if target_season in peak:
            return 0.95

        # Check avoid seasons
        if any(target_season in s for s in avoid_seasons):
            return 0.2

        # Partial/neutral season
        return 0.65

    def _compute_duration_score(self, destination: Dict[str, Any], target_days: Optional[int]) -> float:
        """Evaluates trip duration alignment (0.0 to 1.0)."""
        if not target_days:
            return 1.0

        ideal = destination.get("ideal_days", 4)
        min_d = destination.get("minimum_days", 2)
        max_d = destination.get("maximum_days", 7)

        # Perfect match if within [min_d, max_d]
        if min_d <= target_days <= max_d:
            # Bonus if exact ideal match
            diff = abs(target_days - ideal)
            return max(0.85, 1.0 - (diff * 0.05))

        # Penalty proportional to distance
        if target_days < min_d:
            diff = min_d - target_days
            return max(0.3, 0.8 - (diff * 0.15))
        else:
            diff = target_days - max_d
            return max(0.4, 0.85 - (diff * 0.1))

    def _compute_budget_score(self, destination: Dict[str, Any], target_budget: Optional[float], target_days: Optional[int]) -> float:
        """Evaluates budget feasibility."""
        if not target_budget:
            return 1.0

        days = target_days or destination.get("ideal_days", 4)
        budget_cat = destination.get("budget_category", {})
        daily_range = budget_cat.get("total_daily_range", [1200, 2500])

        min_trip_cost = daily_range[0] * days
        mid_trip_cost = daily_range[1] * days

        # If user budget exceeds minimum daily cost comfortably
        if target_budget >= mid_trip_cost:
            return 1.0
        elif target_budget >= min_trip_cost:
            # Within budget range
            ratio = (target_budget - min_trip_cost) / max(1, (mid_trip_cost - min_trip_cost))
            return 0.8 + (ratio * 0.2)
        else:
            # Below budget range
            deficit_ratio = target_budget / max(1, min_trip_cost)
            return max(0.2, deficit_ratio * 0.7)

    def recommend(self, preferences: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Recommends top K destinations based on user preferences.
        """
        if not self.destinations or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        days = preferences.get("days")
        budget = preferences.get("budget")
        interests = preferences.get("interests", [])
        season = preferences.get("season")
        group_type = preferences.get("group_type")

        # 1. Build Query Text for TF-IDF Vectorization
        query_parts = []
        if interests:
            # Add interest words and semantic synonyms
            for interest in interests:
                query_parts.append(interest)
                if interest == "nature":
                    query_parts.append("scenic hill station greenery mountains landscape")
                elif interest == "adventure":
                    query_parts.append("trekking rafting camping outdoor thrill")
                elif interest == "beach":
                    query_parts.append("coastal sea watersports sand")
                elif interest == "cultural":
                    query_parts.append("heritage history monument palace fort temple")
        if season:
            query_parts.append(season)
        if group_type:
            query_parts.append(group_type)

        query_text = " ".join(query_parts) if query_parts else "popular tourist destination in India"

        # 2. Compute TF-IDF Cosine Similarity
        user_vec = self.vectorizer.transform([query_text])
        raw_sim_scores = cosine_similarity(user_vec, self.tfidf_matrix)[0]

        # Normalize TF-IDF scores between 0 and 1
        max_sim = np.max(raw_sim_scores) if np.max(raw_sim_scores) > 0 else 1.0
        norm_sim_scores = raw_sim_scores / max_sim if max_sim > 0 else raw_sim_scores

        # 3. Calculate Multi-Factor Composite Scores
        scored_destinations = []
        for idx, dest in enumerate(self.destinations):
            sim_score = norm_sim_scores[idx]
            season_score = self._compute_season_score(dest, season)
            duration_score = self._compute_duration_score(dest, days)
            budget_score = self._compute_budget_score(dest, budget, days)
            popularity = (dest.get("popularity_score", 7) / 10.0)

            # Weighted Hybrid Score Formula:
            # 50% Text/Interest Semantic Match + 20% Season + 15% Duration + 10% Budget + 5% Popularity
            composite_score = (
                (sim_score * 0.50) +
                (season_score * 0.20) +
                (duration_score * 0.15) +
                (budget_score * 0.10) +
                (popularity * 0.05)
            )

            # Match Percentage scaled realistically between 60% and 98%
            match_percentage = min(98, max(50, int(composite_score * 100)))

            daily_budget = dest.get("budget_category", {}).get("total_daily_range", [1500, 3000])

            scored_destinations.append({
                "id": dest.get("id", idx + 1),
                "destination_name": dest.get("destination_name"),
                "state": dest.get("state"),
                "region": dest.get("region"),
                "trip_types": dest.get("trip_types", []),
                "best_seasons": dest.get("best_seasons", []),
                "ideal_days": dest.get("ideal_days", 4),
                "minimum_days": dest.get("minimum_days", 2),
                "maximum_days": dest.get("maximum_days", 7),
                "daily_budget_range": daily_budget,
                "primary_attractions": dest.get("primary_attractions", [])[:4],
                "activities_available": dest.get("activities_available", [])[:4],
                "unique_experiences": dest.get("unique_experiences", ""),
                "popularity_score": dest.get("popularity_score", 8),
                "match_percentage": match_percentage,
                "composite_score": composite_score
            })

        # 4. Sort and Rank Top K Destinations
        scored_destinations.sort(key=lambda x: x["composite_score"], reverse=True)
        top_destinations = scored_destinations[:top_k]

        # Assign Rank Badges (🥇, 🥈, 🥉)
        rank_badges = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for i, d in enumerate(top_destinations):
            d["rank"] = i + 1
            d["badge"] = rank_badges[i] if i < len(rank_badges) else f"#{i+1}"

        return top_destinations


# Singleton instance
destination_recommender = DestinationRecommender()
