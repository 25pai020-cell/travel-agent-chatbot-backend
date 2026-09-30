import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from services.recommendation_service import destination_recommender


def test_recommend_nature_adventure_winter():
    pref = {
        "days": 4,
        "budget": 20000.0,
        "interests": ["nature", "adventure"],
        "season": "winter"
    }

    recs = destination_recommender.recommend(pref, top_k=3)
    print("Test 1 - 4-day Nature & Adventure in Winter:")
    for r in recs:
        print(f"  Rank {r['rank']}: {r['destination_name']} ({r['state']}) -> {r['match_percentage']}% Match")
        print(f"     Trip Types: {r['trip_types']}")
        print(f"     Attractions: {r['primary_attractions']}")

    assert len(recs) == 3
    names = [r["destination_name"] for r in recs]
    print(f"  Recommended names: {names}")
    print("[PASS] Test 1: Destination recommendations generated successfully!")


def test_recommend_beach_friends():
    pref = {
        "days": 3,
        "budget": 15000.0,
        "interests": ["beach"],
        "group_type": "friends",
        "season": "winter"
    }

    recs = destination_recommender.recommend(pref, top_k=3)
    print("\nTest 2 - 3-day Beach with Friends in Winter:")
    for r in recs:
        print(f"  Rank {r['rank']}: {r['destination_name']} ({r['state']}) -> {r['match_percentage']}% Match")
    assert len(recs) == 3
    print("[PASS] Test 2: Beach recommendations generated successfully!")


if __name__ == "__main__":
    test_recommend_nature_adventure_winter()
    test_recommend_beach_friends()
    print("\n[SUCCESS] ALL RECOMMENDATION TESTS PASSED PERFECTLY!")
