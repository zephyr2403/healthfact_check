import requests
import pytest

url = "http://127.0.0.1:31313/claim/v1/predict"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}


@pytest.mark.parametrize("claim_text, expected_veracity", [
    ("The media covered up an incident in San Bernardino during which several Muslim men \
      fired upon a number of Californian hikers.",0),
    ("Treating at the Earliest Sign of MS May Offer Long-Term Benefit",1),
    ("Hospitals plan for 'COVID cabanas,' conference rooms to house patients.",2),
    ("Women’s menstrual cycles synchronize when they live or work in close \
    proximity to one another.",3)
])
def test_claim_veracity(claim_text, expected_veracity):
    payload = {"claim_text": claim_text}
    response = requests.post(url, headers=headers, json=payload)
    
    assert response.status_code == 200, f"Expected 200, but got {response.status_code}"
    assert response.json() == {
        "message": "success",
        "veracity": expected_veracity
    }
