import sys
import os
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from services.nlp_service import nlp_extractor


def test_user_main_query():
    text = "I want a 4-day nature and adventure trip under ₹20,000 in winter."
    result = nlp_extractor.extract_preferences(text)
    print("Test 1 Result:", result)

    assert result["days"] == 4, f"Expected 4 days, got {result['days']}"
    assert result["budget"] == 20000.0, f"Expected 20000 budget, got {result['budget']}"
    assert "nature" in result["interests"], f"Expected nature in {result['interests']}"
    assert "adventure" in result["interests"], f"Expected adventure in {result['interests']}"
    assert result["season"] == "winter", f"Expected winter, got {result['season']}"
    print("[PASS] Test 1: User main query passed!")


def test_variation_queries():
    test_cases = [
        {
            "query": "Looking for a weekend beach getaway with friends under 15k",
            "expected_days": 2,
            "expected_budget": 15000.0,
            "expected_interests": ["beach"],
            "expected_group": "friends"
        },
        {
            "query": "Plan a 7-day cultural and heritage pilgrimage tour in autumn for family with budget 50000 inr",
            "expected_days": 7,
            "expected_budget": 50000.0,
            "expected_interests": ["cultural", "religious"],
            "expected_season": "post-monsoon",
            "expected_group": "family"
        },
        {
            "query": "I want a solo 5 days wildlife safari and photography trip in December under 30000",
            "expected_days": 5,
            "expected_budget": 30000.0,
            "expected_interests": ["photography", "wildlife"],
            "expected_season": "winter",
            "expected_group": "solo"
        }
    ]

    for i, tc in enumerate(test_cases, 2):
        res = nlp_extractor.extract_preferences(tc["query"])
        print(f"Test {i} Query: {tc['query']}")
        print(f"Test {i} Result: {res}")
        if "expected_days" in tc:
            assert res["days"] == tc["expected_days"]
        if "expected_budget" in tc:
            assert res["budget"] == tc["expected_budget"]
        if "expected_group" in tc:
            assert res["group_type"] == tc["expected_group"]
        if "expected_season" in tc:
            assert res["season"] == tc["expected_season"]
        print(f"[PASS] Test {i} Passed!")


if __name__ == "__main__":
    test_user_main_query()
    test_variation_queries()
    print("\n[SUCCESS] ALL NLP TESTS PASSED PERFECTLY!")
