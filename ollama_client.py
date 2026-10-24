"""
Ollama Local AI Client for Airbnb Listing Enhancement.
Connects to the user's local Ollama instance (100% free, private, offline).
Complies with GDPR and local data privacy: zero external requests.
"""

import os
import json
import urllib.request
import urllib.error

OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434/api")

class OllamaAdvisor:
    def __init__(self, base_url: str = OLLAMA_API_BASE):
        self.base_url = base_url

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is active and responding."""
        try:
            req = urllib.request.Request(f"{self.base_url}/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_available_models(self) -> list:
        """Returns list of installed model names in Ollama."""
        try:
            req = urllib.request.Request(f"{self.base_url}/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                return models
        except Exception:
            return []

    def generate_listing_advice(self, listing_data: dict, predicted_price: float, model_name: str = "llama3.2:latest") -> str:
        """
        Queries local Ollama to provide intelligent, tailored pricing suggestions
        and advice on maximizing occupancy and earnings.
        """
        missing_amenities = []
        amenity_labels = {
            "has_air_conditioning": "Air Conditioning",
            "has_free_parking": "Free Parking",
            "has_workspace": "Dedicated Workspace",
            "has_balcony": "Balcony / Terrace",
            "has_washer": "Washing Machine",
            "is_pet_friendly": "Pet Friendly Policy",
            "has_pool_or_hot_tub": "Hot Tub / Pool"
        }
        for k, label in amenity_labels.items():
            if not listing_data.get(k, 0):
                missing_amenities.append(label)

        prompt = f"""You are an expert Airbnb revenue management consultant in Germany.
Analyze this property and its machine-learning estimated nightly price:
- City: {listing_data.get('city')} ({listing_data.get('neighborhood_type')})
- Property Type: {listing_data.get('room_type')}
- Capacity: {listing_data.get('accommodates')} guests, {listing_data.get('bedrooms')} bedrooms, {listing_data.get('bathrooms')} baths
- Estimated Base Price: €{predicted_price:.2f}/night
- Current Rating: {listing_data.get('review_scores_rating')}/5.0 ({listing_data.get('number_of_reviews')} reviews)
- High-value amenities currently missing: {', '.join(missing_amenities[:4]) if missing_amenities else 'None (Fully equipped)'}

In 3 concise bullet points:
1. Pricing Strategy: Suggest an optimal weekday vs weekend or seasonal pricing range.
2. Value Addition: Which single upgrade or amenity would yield the highest ROI here?
3. Host Tip: A quick recommendation to maintain superhost status or maximize bookings.

Keep the response brief, professional, and directly actionable."""

        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "num_predict": 250
                }
            }
            req = urllib.request.Request(
                f"{self.base_url}/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=35.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return res_data.get("response", "").strip()
        except Exception as e:
            return (
                f"**Built-in Quick Tip (Local Ollama offline or busy):**\n"
                f"- **Recommended Price Range:** €{predicted_price * 0.95:.0f} (midweek) – €{predicted_price * 1.15:.0f} (peak weekends).\n"
                f"- **Quick Upgrade:** Adding a dedicated laptop workspace or flexible self check-in can boost bookings by up to 15%.\n"
                f"- **Local Tip:** Properties in {listing_data.get('city')} see strong demand during local conventions and cultural weekends."
            )

    def generate_listing_title_and_description(self, listing_data: dict, model_name: str = "llama3.2:latest") -> dict:
        """Generates an attractive headline and 2-sentence listing description."""
        prompt = f"""Write an attractive Airbnb title (max 8 words) and a short 2-sentence listing summary for this rental:
City: {listing_data.get('city')}
Neighborhood: {listing_data.get('neighborhood_type')}
Type: {listing_data.get('room_type')}
Guests: {listing_data.get('accommodates')}
Bedrooms: {listing_data.get('bedrooms')}

Respond ONLY in JSON format:
{{"title": "...", "summary": "..."}}"""

        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.7,
                    "num_predict": 120
                }
            }
            req = urllib.request.Request(
                f"{self.base_url}/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                text = res_data.get("response", "{}")
                parsed = json.loads(text)
                return parsed
        except Exception:
            return {
                "title": f"Charming {listing_data.get('room_type')} in {listing_data.get('city')}",
                "summary": f"Enjoy a comfortable stay in {listing_data.get('city')}, perfectly positioned in {listing_data.get('neighborhood_type')}."
            }
