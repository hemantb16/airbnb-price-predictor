"""
Generate a realistic, high-quality Airbnb dataset for German cities.
Includes realistic price distributions, coordinates, room types, and amenities.
Compliant with local privacy rules (fully synthetic benchmark data).
"""

import numpy as np
import pandas as pd
import os

# City center coordinates and base price multipliers
CITIES = {
    "Munich": {"lat": 48.1351, "lon": 11.5820, "base_price": 115, "spread": 0.05},
    "Berlin": {"lat": 52.5200, "lon": 13.4050, "base_price": 88, "spread": 0.07},
    "Hamburg": {"lat": 53.5511, "lon": 9.9937, "base_price": 92, "spread": 0.06},
    "Frankfurt": {"lat": 50.1109, "lon": 8.6821, "base_price": 105, "spread": 0.05},
    "Cologne": {"lat": 50.9375, "lon": 6.9603, "base_price": 82, "spread": 0.05},
    "Stuttgart": {"lat": 48.7758, "lon": 9.1829, "base_price": 85, "spread": 0.05},
    "Dresden": {"lat": 51.0504, "lon": 13.7373, "base_price": 68, "spread": 0.05},
    "Leipzig": {"lat": 51.3397, "lon": 12.3731, "base_price": 64, "spread": 0.05},
}

NEIGHBORHOOD_TYPES = {
    "City Center / Altstadt": 1.35,
    "Trendy / Cultural": 1.15,
    "Residential / Quiet": 0.95,
    "Green / Lakes / Nature": 1.05,
    "Suburban / Outskirts": 0.80,
}

ROOM_TYPES = {
    "Entire home/apt": 1.45,
    "Private room": 0.70,
    "Hotel room": 1.25,
    "Shared room": 0.38,
}

def generate_airbnb_data(n_samples=4000, random_seed=42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    city_names = list(CITIES.keys())
    # Population-weighted distribution for cities
    city_weights = [0.22, 0.28, 0.16, 0.12, 0.10, 0.05, 0.04, 0.03]
    city_choices = np.random.choice(city_names, size=n_samples, p=city_weights)
    
    neighborhood_choices = np.random.choice(
        list(NEIGHBORHOOD_TYPES.keys()),
        size=n_samples,
        p=[0.30, 0.25, 0.25, 0.10, 0.10]
    )
    
    room_choices = np.random.choice(
        list(ROOM_TYPES.keys()),
        size=n_samples,
        p=[0.68, 0.28, 0.02, 0.02]
    )
    
    # Property attributes correlated with room type
    accommodates = []
    bedrooms = []
    bathrooms = []
    beds = []
    
    for rt in room_choices:
        if rt == "Entire home/apt":
            acc = int(np.clip(np.random.choice([1, 2, 3, 4, 5, 6, 8], p=[0.10, 0.35, 0.15, 0.22, 0.08, 0.07, 0.03]), 1, 10))
            bed = int(np.clip(round(acc / 2 + np.random.choice([-0.5, 0, 0.5])), 1, 5))
            bath = float(np.clip(round(1.0 + (acc > 4) * 0.5 + (acc > 6) * 0.5 + np.random.choice([0, 0.5], p=[0.7, 0.3])), 1.0, 4.0))
            b_count = int(np.clip(round(acc * 0.75), 1, 8))
        elif rt == "Private room":
            acc = int(np.random.choice([1, 2, 3], p=[0.45, 0.45, 0.10]))
            bed = 1
            bath = float(np.random.choice([1.0, 1.5, 2.0], p=[0.75, 0.15, 0.10]))
            b_count = int(np.random.choice([1, 2], p=[0.75, 0.25]))
        elif rt == "Hotel room":
            acc = int(np.random.choice([1, 2, 3], p=[0.3, 0.6, 0.1]))
            bed = 1
            bath = 1.0
            b_count = int(np.random.choice([1, 2], p=[0.7, 0.3]))
        else: # Shared room
            acc = 1
            bed = 1
            bath = 1.0
            b_count = 1
            
        accommodates.append(acc)
        bedrooms.append(bed)
        bathrooms.append(bath)
        beds.append(b_count)
        
    accommodates = np.array(accommodates)
    bedrooms = np.array(bedrooms)
    bathrooms = np.array(bathrooms)
    beds = np.array(beds)
    
    # Amenities probabilities
    has_wifi = np.random.binomial(1, 0.96, n_samples)
    has_kitchen = np.where(room_choices == "Shared room", np.random.binomial(1, 0.4, n_samples), np.random.binomial(1, 0.90, n_samples))
    has_air_conditioning = np.random.binomial(1, 0.28, n_samples)
    has_free_parking = np.random.binomial(1, 0.38, n_samples)
    has_washer = np.random.binomial(1, 0.72, n_samples)
    has_workspace = np.random.binomial(1, 0.65, n_samples)
    has_balcony = np.random.binomial(1, 0.44, n_samples)
    has_elevator = np.random.binomial(1, 0.52, n_samples)
    is_pet_friendly = np.random.binomial(1, 0.24, n_samples)
    has_pool_or_hot_tub = np.random.binomial(1, 0.08, n_samples)
    
    minimum_nights = np.random.choice([1, 2, 3, 4, 5, 7, 14, 30], size=n_samples, p=[0.30, 0.35, 0.18, 0.07, 0.04, 0.04, 0.01, 0.01])
    number_of_reviews = np.random.exponential(scale=35, size=n_samples).astype(int)
    number_of_reviews = np.clip(number_of_reviews, 0, 480)
    
    review_scores_rating = np.random.beta(a=12, b=1.5, size=n_samples) * 2.0 + 3.0
    review_scores_rating = np.round(np.clip(review_scores_rating, 3.2, 5.0), 2)
    
    # Generate realistic latitude and longitude around city centers
    lats = []
    lons = []
    for city in city_choices:
        base_lat = CITIES[city]["lat"]
        base_lon = CITIES[city]["lon"]
        spread = CITIES[city]["spread"]
        lats.append(base_lat + np.random.normal(0, spread * 0.4))
        lons.append(base_lon + np.random.normal(0, spread * 0.4))
        
    # Calculate price based on underlying factors + realistic noise
    prices = []
    for i in range(n_samples):
        city = city_choices[i]
        nh = neighborhood_choices[i]
        rt = room_choices[i]
        
        base = CITIES[city]["base_price"]
        nh_factor = NEIGHBORHOOD_TYPES[nh]
        rt_factor = ROOM_TYPES[rt]
        
        # Room size and capacity contributions
        capacity_boost = (accommodates[i] - 1) * 16.0 + (bedrooms[i] - 1) * 22.0 + (bathrooms[i] - 1.0) * 18.0
        
        # Amenities premium
        amenity_val = (
            has_wifi[i] * 5.0 +
            has_kitchen[i] * 12.0 +
            has_air_conditioning[i] * 24.0 +
            has_free_parking[i] * 16.0 +
            has_washer[i] * 8.0 +
            has_workspace[i] * 9.0 +
            has_balcony[i] * 14.0 +
            has_elevator[i] * 7.0 +
            is_pet_friendly[i] * 6.0 +
            has_pool_or_hot_tub[i] * 45.0
        )
        
        # Review boost (high ratings command small premium)
        rating_boost = max(0, (review_scores_rating[i] - 4.5) * 18.0)
        
        # Minimum nights discount for long stays
        stay_discount = 0.92 if minimum_nights[i] >= 7 else 1.0
        
        # Target price formula
        price = (base * nh_factor * rt_factor + capacity_boost + amenity_val + rating_boost) * stay_discount
        
        # Add realistic multiplicative log-normal noise
        noise = np.random.lognormal(mean=0, sigma=0.15)
        final_price = round(float(price * noise), 2)
        final_price = max(18.0, min(final_price, 750.0))
        prices.append(final_price)
        
    df = pd.DataFrame({
        "city": city_choices,
        "neighborhood_type": neighborhood_choices,
        "room_type": room_choices,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "beds": beds,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "review_scores_rating": review_scores_rating,
        "has_wifi": has_wifi,
        "has_kitchen": has_kitchen,
        "has_air_conditioning": has_air_conditioning,
        "has_free_parking": has_free_parking,
        "has_washer": has_washer,
        "has_workspace": has_workspace,
        "has_balcony": has_balcony,
        "has_elevator": has_elevator,
        "is_pet_friendly": is_pet_friendly,
        "has_pool_or_hot_tub": has_pool_or_hot_tub,
        "latitude": np.round(lats, 5),
        "longitude": np.round(lons, 5),
        "price": prices
    })
    
    return df

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "airbnb_listings.csv")
    
    print("Generating realistic Airbnb dataset...")
    df = generate_airbnb_data(n_samples=4000)
    df.to_csv(output_file, index=False)
    print(f"Dataset successfully created with {len(df)} listings at: {output_file}")
    print(df.head())
    print("\nPrice statistics (EUR):")
    print(df["price"].describe())
