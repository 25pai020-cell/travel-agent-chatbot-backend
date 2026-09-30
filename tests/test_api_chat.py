import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from fastapi.testclient import TestClient
from app import app


def test_chat_message_flow():
    client = TestClient(app)
    res = client.post('/api/chat/message', json={'message': 'I want a 4-day nature and adventure trip under Rs 20000 in winter.'})
    print('Status:', res.status_code)
    assert res.status_code == 200
    data = res.json()
    print('Extracted Days:', data['extracted_preferences']['days'])
    print('Extracted Interests:', data['extracted_preferences']['interests'])
    print('Extracted Season:', data['extracted_preferences']['season'])
    print('Destinations count:', len(data['destinations']))

    for d in data['destinations']:
        print(f"Rank {d['rank']}: {d['destination_name']} ({d['state']}) -> {d['match_percentage']}% Match")

    assert len(data['destinations']) == 3
    assert data['destinations'][0]['destination_name'] == 'Manali'
    print('[PASS] End-to-end Chat Endpoint returns NLP preferences + ML destinations successfully!')


if __name__ == '__main__':
    test_chat_message_flow()
    print('\n[SUCCESS] API CHAT FLOW TEST PASSED!')
