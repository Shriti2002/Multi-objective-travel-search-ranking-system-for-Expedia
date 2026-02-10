"""
Data utilities for Travel Search Ranking System
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def generate_synthetic_queries(num_queries: int = 10000, 
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic hotel search queries with intent labels
    
    Returns:
        DataFrame with columns: query_id, query_text, intent_labels (list)
    """
    np.random.seed(seed)
    
    # Query templates
    locations = ["seattle", "new york", "miami", "san francisco", "chicago", 
                 "boston", "las vegas", "orlando", "los angeles", "austin"]
    
    landmarks = ["airport", "downtown", "beach", "convention center", 
                 "times square", "disneyland", "strip"]
    
    price_terms = ["cheap", "budget", "affordable", "luxury", "5-star", 
                   "upscale", "expensive", "premium"]
    
    amenities = ["pool", "spa", "gym", "restaurant", "shuttle", "parking",
                 "wifi", "breakfast", "bar"]
    
    purposes = ["family", "business", "romantic", "vacation", "conference",
                "wedding", "honeymoon"]
    
    modifiers = ["near", "close to", "walking distance to", "with", "and"]
    
    queries = []
    
    for i in range(num_queries):
        query_parts = []
        intents = []
        
        # Price intent (30% of queries)
        if np.random.random() < 0.3:
            price_term = np.random.choice(price_terms)
            query_parts.append(price_term)
            if price_term in ["cheap", "budget", "affordable"]:
                intents.append("budget")
            elif price_term in ["luxury", "5-star", "upscale", "expensive", "premium"]:
                intents.append("luxury")
        
        # Always include "hotel"
        query_parts.append("hotel")
        
        # Location (80% of queries)
        if np.random.random() < 0.8:
            location = np.random.choice(locations)
            if np.random.random() < 0.4:  # 40% include landmark
                landmark = np.random.choice(landmarks)
                query_parts.extend([np.random.choice(modifiers), landmark])
                
                # Add landmark-specific intents
                if landmark == "airport":
                    intents.append("airport")
                elif landmark in ["downtown", "times square", "strip"]:
                    intents.append("downtown")
                elif landmark == "beach":
                    intents.append("beach")
            else:
                query_parts.extend(["in", location])
        
        # Purpose (20% of queries)
        if np.random.random() < 0.2:
            purpose = np.random.choice(purposes)
            query_parts.extend(["for", purpose])
            if purpose in ["family", "business", "romantic"]:
                intents.append(purpose)
        
        # Amenities (40% of queries)
        if np.random.random() < 0.4:
            num_amenities = np.random.randint(1, 3)
            selected_amenities = np.random.choice(amenities, num_amenities, replace=False)
            query_parts.extend(["with", " and ".join(selected_amenities)])
        
        # If no intent identified, assign generic
        if len(intents) == 0:
            intents.append("business")  # default
        
        query_text = " ".join(query_parts)
        
        queries.append({
            'query_id': f"q_{i:06d}",
            'query_text': query_text,
            'intent_labels': intents
        })
    
    return pd.DataFrame(queries)


def generate_synthetic_hotels(num_hotels: int = 5000, 
                              seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic hotel dataset
    
    Returns:
        DataFrame with hotel features
    """
    np.random.seed(seed)
    
    hotels = []
    
    for i in range(num_hotels):
        # Price distribution (log-normal)
        base_price = np.random.lognormal(mean=4.8, sigma=0.5)  # ~$120-$400
        
        # Rating (skewed towards higher ratings)
        rating = np.clip(np.random.beta(8, 2) * 5, 1, 5)
        
        # Reviews (correlated with rating)
        num_reviews = int(np.random.exponential(scale=200) * (rating / 3.0))
        
        # Distance from city center (exponential)
        distance_km = np.random.exponential(scale=5)
        
        # Amenities (more for higher price)
        amenities_count = int(np.clip(
            np.random.poisson(lam=5 + (base_price / 50)), 
            2, 15
        ))
        
        # Cancellation flexibility (random)
        cancellation_flexibility = np.random.choice(
            ['free', 'partial', 'strict'], 
            p=[0.4, 0.4, 0.2]
        )
        
        # Room types
        room_types_count = np.random.randint(2, 8)
        
        hotels.append({
            'hotel_id': f"h_{i:06d}",
            'hotel_name': f"Hotel_{i:04d}",
            'hotel_price': round(base_price, 2),
            'hotel_rating': round(rating, 2),
            'num_reviews': num_reviews,
            'distance_km': round(distance_km, 2),
            'amenities_count': amenities_count,
            'cancellation_flexibility': cancellation_flexibility,
            'room_types_count': room_types_count,
            'hotel_description': f"A {'luxury' if base_price > 200 else 'comfortable'} hotel with {amenities_count} amenities"
        })
    
    return pd.DataFrame(hotels)


def generate_query_hotel_pairs(queries_df: pd.DataFrame,
                               hotels_df: pd.DataFrame,
                               avg_candidates: int = 100,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate query-hotel candidate pairs with synthetic labels
    
    Uses a behavior model to create realistic click/book probabilities
    """
    np.random.seed(seed)
    
    config = load_config()
    click_params = config['synthetic']['click_model']
    book_params = config['synthetic']['book_model']
    
    pairs = []
    
    for _, query in queries_df.iterrows():
        # Sample candidates for this query
        num_candidates = np.random.poisson(avg_candidates)
        num_candidates = min(num_candidates, len(hotels_df))
        
        candidate_hotels = hotels_df.sample(n=num_candidates)
        
        for _, hotel in candidate_hotels.iterrows():
            # Compute synthetic features
            
            # Price competitiveness (normalized)
            price_percentile = (hotel['hotel_price'] - hotels_df['hotel_price'].min()) / \
                             (hotels_df['hotel_price'].max() - hotels_df['hotel_price'].min())
            
            # Intent match score (synthetic)
            intent_match = 0.5  # baseline
            if 'budget' in query['intent_labels'] and hotel['hotel_price'] < 150:
                intent_match += 0.3
            if 'luxury' in query['intent_labels'] and hotel['hotel_price'] > 200:
                intent_match += 0.3
            if 'airport' in query['intent_labels'] and hotel['distance_km'] < 5:
                intent_match += 0.2
            intent_match = min(intent_match, 1.0)
            
            # Click probability (behavior model)
            click_logit = (
                click_params['base_rate'] +
                click_params['price_sensitivity'] * price_percentile +
                click_params['rating_boost'] * (hotel['hotel_rating'] / 5.0) +
                click_params['intent_match_boost'] * intent_match +
                click_params['distance_penalty'] * min(hotel['distance_km'] / 10.0, 1.0) +
                np.random.normal(0, click_params['noise_std'])
            )
            click_prob = 1 / (1 + np.exp(-click_logit))
            clicked = int(np.random.random() < click_prob)
            
            # Book probability (conditional on click)
            if clicked:
                book_logit = (
                    book_params['base_rate'] +
                    book_params['cancellation_boost'] * (1 if hotel['cancellation_flexibility'] == 'free' else 0) +
                    book_params['amenities_boost'] * (hotel['amenities_count'] / 15.0) +
                    book_params['reviews_boost'] * min(hotel['num_reviews'] / 500, 1.0) +
                    np.random.normal(0, book_params['noise_std'])
                )
                book_prob = 1 / (1 + np.exp(-book_logit))
                booked = int(np.random.random() < book_prob)
            else:
                booked = 0
                book_prob = 0.0
            
            # Relevance score (0-4 scale for ranking)
            relevance = int(clicked * 2 + booked * 2)  # 0, 2, or 4
            
            pairs.append({
                'query_id': query['query_id'],
                'query_text': query['query_text'],
                'hotel_id': hotel['hotel_id'],
                'hotel_price': hotel['hotel_price'],
                'hotel_rating': hotel['hotel_rating'],
                'distance_km': hotel['distance_km'],
                'num_reviews': hotel['num_reviews'],
                'amenities_count': hotel['amenities_count'],
                'cancellation_flexibility': hotel['cancellation_flexibility'],
                'room_types_count': hotel['room_types_count'],
                'intent_match_score': intent_match,
                'price_competitiveness': 1 - price_percentile,
                'clicked': clicked,
                'booked': booked,
                'click_prob': click_prob,
                'book_prob': book_prob,
                'relevance': relevance
            })
    
    return pd.DataFrame(pairs)


def create_train_test_split(df: pd.DataFrame, 
                            test_size: float = 0.2,
                            group_col: str = 'query_id',
                            seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by groups (queries) to avoid leakage
    """
    np.random.seed(seed)
    
    unique_groups = df[group_col].unique()
    np.random.shuffle(unique_groups)
    
    split_idx = int(len(unique_groups) * (1 - test_size))
    train_groups = unique_groups[:split_idx]
    test_groups = unique_groups[split_idx:]
    
    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    
    return train_df, test_df


def save_data(df: pd.DataFrame, filename: str, data_dir: str = "data/processed"):
    """Save DataFrame to parquet"""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(data_dir) / filename
    df.to_parquet(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")


def load_data(filename: str, data_dir: str = "data/processed") -> pd.DataFrame:
    """Load DataFrame from parquet"""
    filepath = Path(data_dir) / filename
    df = pd.read_parquet(filepath)
    print(f"Loaded {len(df)} rows from {filepath}")
    return df
