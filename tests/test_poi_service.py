import unittest
import sys
from pathlib import Path

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.poi_service import poi_service
from services.nlp_service import nlp_extractor


class TestPOIService(unittest.TestCase):

    def test_manali_pois_retrieval(self):
        """Test retrieving POIs for Manali from local datasets."""
        pois = poi_service.get_pois_for_destination("Manali", user_interests=[], limit=10)
        self.assertTrue(len(pois) > 0, "Should retrieve POIs for Manali")
        
        poi_names = [p["name"] for p in pois]
        # Check canonical Manali POIs are present
        self.assertTrue(any("Solang" in name for name in poi_names), "Solang Valley should be found")
        self.assertTrue(any("Hadimba" in name or "Hidimba" in name for name in poi_names), "Hadimba Temple should be found")
        self.assertTrue(any("Jogini" in name for name in poi_names), "Jogini Falls should be found")
        self.assertTrue(any("Mall Road" in name or "Market" in name for name in poi_names), "Mall Road should be found")

    def test_poi_characteristics_and_high_match(self):
        """Test POI characteristics enrichment and HIGH MATCH matching for Nature + Adventure."""
        pois = poi_service.get_pois_for_destination("Manali", user_interests=["nature", "adventure"], limit=10)
        
        solang = next((p for p in pois if "Solang" in p["name"]), None)
        self.assertIsNotNone(solang, "Solang Valley POI should exist")
        
        # Verify characteristics
        chars_lower = [c.lower() for c in solang["characteristics"]]
        self.assertTrue("nature" in chars_lower or "adventure" in chars_lower or "snow" in chars_lower, "Should have characteristic tags")
        
        # Verify HIGH MATCH status
        self.assertEqual(solang["match_level"], "HIGH MATCH", "Solang Valley should be a HIGH MATCH for nature + adventure")
        self.assertTrue(solang["is_high_match"], "is_high_match flag should be True")
        self.assertGreaterEqual(solang["match_score"], 80, "Match score should be >= 80%")


    def test_destination_nlp_extraction(self):
        """Test extracting selected destination from user query."""
        test_cases = [
            ("User: Manali", "Manali"),
            ("Selected Destination = Manali", "Manali"),
            ("I choose Manali! Build my itinerary", "Manali"),
            ("Let's go to Munnar", "Munnar"),
            ("Goa", "Goa")
        ]
        for query, expected_dest in test_cases:
            pref = nlp_extractor.extract_preferences(query)
            self.assertEqual(pref.get("destination"), expected_dest, f"Failed to extract {expected_dest} from '{query}'")

    def test_poi_scoring_breakdown(self):
        """Test exact 100-point scoring algorithm (Interest Match 40, Rating 25, Characteristics 25, Season 10)."""
        pois = poi_service.get_pois_for_destination("Manali", user_interests=["nature", "adventure"], limit=10)
        solang = next((p for p in pois if "Solang" in p["name"]), None)
        self.assertIsNotNone(solang, "Solang Valley POI should exist")

        self.assertIn("total_score", solang, "POI should have total_score")
        self.assertIn("interest_match_score", solang, "POI should have interest_match_score")
        self.assertIn("rating_score", solang, "POI should have rating_score")
        self.assertIn("characteristics_score", solang, "POI should have characteristics_score")
        self.assertIn("season_score", solang, "POI should have season_score")

        # Solang Valley should get close to 100 score for nature + adventure
        self.assertGreaterEqual(solang["total_score"], 85.0, "Solang Valley should have high score for nature + adventure")

    def test_custom_itinerary_algorithm_and_constraints(self):
        """Test Custom Itinerary Algorithm: Max 8 hours/day and Budget check <= 20,000."""
        itinerary = poi_service.generate_custom_itinerary(
            destination_name="Manali",
            days=4,
            budget=20000.0,
            user_interests=["nature", "adventure"]
        )

        self.assertEqual(itinerary["destination"], "Manali")
        self.assertEqual(itinerary["days"], 4)
        self.assertTrue(itinerary["within_budget"], "Total cost should be within budget")
        self.assertLessEqual(itinerary["total_cost"], 20000.0, "Total cost must not exceed ₹20,000")

        # Verify daily 8-hour limit
        for day_plan in itinerary["day_plans"]:
            self.assertLessEqual(day_plan["total_time_hrs"], 8.0, f"Day {day_plan['day']} sightseeing hours must be <= 8.0")
            self.assertTrue(len(day_plan["pois"]) > 0, f"Day {day_plan['day']} should have POIs assigned")

    def test_itinerary_modification_remove_temples(self):
        """Test 1️⃣4️⃣ User Modification: 'Remove temples and add more adventure'."""
        mod = nlp_extractor.extract_preferences("Remove temples and add more adventure")
        self.assertTrue(mod["is_modification"], "Should detect modification intent")
        self.assertIn("temple", mod["remove_categories"], "Should detect temple in remove_categories")

        itinerary = poi_service.generate_custom_itinerary(
            destination_name="Manali",
            days=4,
            budget=20000.0,
            user_interests=["nature", "adventure"],
            removed_categories=mod["remove_categories"]
        )

        all_assigned_names = []
        for day_plan in itinerary["day_plans"]:
            for p in day_plan["pois"]:
                all_assigned_names.append(p["name"].lower())

        self.assertFalse(any("hadimba" in name for name in all_assigned_names), "Hadimba Temple should be removed after request")

    def test_ten_day_ten_thousand_itinerary(self):
        """Test generating 10-day itinerary with ₹10,000 budget: ensure all 10 days are populated with places, times and costs."""
        pref = nlp_extractor.extract_preferences("10 days 10000 rupiess")
        self.assertEqual(pref.get("days"), 10, "Should extract 10 days")
        self.assertEqual(pref.get("budget"), 10000.0, "Should extract 10000 budget")

        itinerary = poi_service.generate_custom_itinerary(
            destination_name="Manali",
            days=10,
            budget=10000.0,
            user_interests=["nature", "scenic"]
        )

        self.assertEqual(len(itinerary["day_plans"]), 10, "Should have exactly 10 day plans")
        for day_plan in itinerary["day_plans"]:
            self.assertGreaterEqual(len(day_plan["pois"]), 3, f"Day {day_plan['day']} must have at least 3 activities (Morning, Afternoon, Evening)")
            self.assertLessEqual(day_plan["total_time_hrs"], 8.0, f"Day {day_plan['day']} total hours must be <= 8.0")
            for poi in day_plan["pois"]:
                self.assertIn("time_slot", poi, "POI should have assigned time_slot")
                self.assertIn("entrance_fee_inr", poi, "POI should have entrance_fee_inr")

        self.assertTrue(itinerary["within_budget"], "Total cost should be within budget")


if __name__ == "__main__":
    unittest.main()

