"""
Feature Engineering for Travel Search Ranking System
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngineer:
    """Feature engineering pipeline for ranking"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.fitted = False
        
    def create_structured_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create structured features from hotel data"""
        features = df.copy()
        
        # Price features
        features['price_log'] = np.log1p(features['hotel_price'])
        features['price_per_rating'] = features['hotel_price'] / (features['hotel_rating'] + 0.1)
        
        # Rating features
        features['rating_squared'] = features['hotel_rating'] ** 2
        features['high_rating'] = (features['hotel_rating'] >= 4.0).astype(int)
        
        # Review features
        features['reviews_log'] = np.log1p(features['num_reviews'])
        features['review_velocity'] = features['num_reviews'] / (features['hotel_rating'] + 0.1)
        
        # Distance features
        features['distance_log'] = np.log1p(features['distance_km'])
        features['is_nearby'] = (features['distance_km'] <= 2.0).astype(int)
        
        # Amenities features
        features['amenities_per_price'] = features['amenities_count'] / (features['hotel_price'] / 100)
        
        # Interaction features
        features['quality_score'] = features['hotel_rating'] * np.log1p(features['num_reviews'])
        features['value_score'] = features['hotel_rating'] / (features['hotel_price'] / 100)
        
        return features
    
    def create_nlp_features(self, df: pd.DataFrame, 
                           query_embeddings: np.ndarray = None,
                           hotel_embeddings: np.ndarray = None) -> pd.DataFrame:
        """
        Create NLP features
        
        Args:
            query_embeddings: Semantic embeddings for queries (shape: [n_queries, embed_dim])
            hotel_embeddings: Semantic embeddings for hotels (shape: [n_hotels, embed_dim])
        """
        features = df.copy()
        
        # Intent match score (already in data)
        # features['intent_match_score'] = ...
        
        # Keyword overlap (simple)
        if 'query_text' in features.columns:
            features['query_length'] = features['query_text'].str.split().str.len()
            features['has_price_term'] = features['query_text'].str.contains(
                'cheap|budget|luxury|expensive', case=False, na=False
            ).astype(int)
            features['has_location_term'] = features['query_text'].str.contains(
                'near|downtown|airport|beach', case=False, na=False
            ).astype(int)
        
        # Semantic similarity (if embeddings provided)
        if query_embeddings is not None and hotel_embeddings is not None:
            # Compute cosine similarity
            # This is simplified - in practice, you'd match query-hotel pairs
            features['semantic_similarity'] = 0.5  # placeholder
        
        return features
    
    def create_contextual_features(self, df: pd.DataFrame, 
                                   add_synthetic: bool = True) -> pd.DataFrame:
        """Create contextual features (session/user context)"""
        features = df.copy()
        
        if add_synthetic:
            # Simulate contextual features
            n = len(features)
            
            features['device_type'] = np.random.choice(
                ['mobile', 'desktop', 'tablet'], 
                size=n, 
                p=[0.6, 0.35, 0.05]
            )
            
            features['party_size'] = np.random.choice(
                [1, 2, 3, 4, 5], 
                size=n,
                p=[0.3, 0.4, 0.15, 0.10, 0.05]
            )
            
            features['length_of_stay'] = np.random.choice(
                [1, 2, 3, 4, 5, 7, 14],
                size=n,
                p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.07, 0.03]
            )
            
            features['booking_window_days'] = np.random.exponential(scale=30, size=n).astype(int)
            features['booking_window_days'] = np.clip(features['booking_window_days'], 0, 365)
            
            features['is_weekend'] = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
            
            features['season'] = np.random.choice(
                ['spring', 'summer', 'fall', 'winter'],
                size=n
            )
        
        # Encode categoricals
        if 'device_type' in features.columns:
            features['device_mobile'] = (features['device_type'] == 'mobile').astype(int)
            features['device_desktop'] = (features['device_type'] == 'desktop').astype(int)
        
        if 'season' in features.columns:
            features['season_summer'] = (features['season'] == 'summer').astype(int)
            features['season_winter'] = (features['season'] == 'winter').astype(int)
        
        return features
    
    def encode_categorical(self, df: pd.DataFrame, 
                          columns: List[str],
                          fit: bool = False) -> pd.DataFrame:
        """Encode categorical variables"""
        features = df.copy()
        
        for col in columns:
            if col not in features.columns:
                continue
                
            if fit:
                self.encoders[col] = LabelEncoder()
                features[f'{col}_encoded'] = self.encoders[col].fit_transform(
                    features[col].astype(str)
                )
            else:
                if col in self.encoders:
                    # Handle unseen labels
                    features[f'{col}_encoded'] = features[col].astype(str).map(
                        lambda x: self.encoders[col].transform([x])[0] 
                        if x in self.encoders[col].classes_ else -1
                    )
        
        return features
    
    def scale_features(self, df: pd.DataFrame,
                      numerical_cols: List[str],
                      fit: bool = False) -> pd.DataFrame:
        """Standardize numerical features"""
        features = df.copy()
        
        if fit:
            self.scaler = StandardScaler()
            features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        else:
            if hasattr(self, 'scaler'):
                features[numerical_cols] = self.scaler.transform(features[numerical_cols])
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of all feature names for ranking"""
        features = [
            # Structured features
            'hotel_price', 'price_log', 'price_per_rating', 'price_competitiveness',
            'hotel_rating', 'rating_squared', 'high_rating',
            'num_reviews', 'reviews_log', 'review_velocity',
            'distance_km', 'distance_log', 'is_nearby',
            'amenities_count', 'amenities_per_price',
            'room_types_count',
            'quality_score', 'value_score',
            
            # NLP features
            'intent_match_score',
            'query_length',
            'has_price_term',
            'has_location_term',
            
            # Contextual features
            'device_mobile', 'device_desktop',
            'party_size', 'length_of_stay',
            'booking_window_days', 'is_weekend',
            'season_summer', 'season_winter',
            
            # Categorical encoded
            'cancellation_flexibility_encoded',
        ]
        
        return features
    
    def transform(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Full feature engineering pipeline"""
        
        # Create all features
        df = self.create_structured_features(df)
        df = self.create_nlp_features(df)
        df = self.create_contextual_features(df, add_synthetic=True)
        
        # Encode categoricals
        categorical_cols = ['cancellation_flexibility']
        df = self.encode_categorical(df, categorical_cols, fit=fit)
        
        # Note: We typically don't scale for tree-based models (XGBoost/LightGBM)
        # But we'll keep the capability
        
        if fit:
            self.fitted = True
        
        return df


def create_ranking_dataset(df: pd.DataFrame,
                          feature_cols: List[str],
                          label_col: str = 'relevance',
                          group_col: str = 'query_id') -> tuple:
    """
    Create dataset in ranking format
    
    Returns:
        X: Feature matrix
        y: Labels (relevance scores)
        groups: Group sizes for ranking
    """
    
    # Sort by group
    df = df.sort_values(group_col).reset_index(drop=True)
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df[label_col].values
    
    # Compute group sizes
    groups = df.groupby(group_col).size().values
    
    return X, y, groups


def add_position_features(df: pd.DataFrame, 
                         group_col: str = 'query_id',
                         score_col: str = 'predicted_score') -> pd.DataFrame:
    """Add ranking position as a feature (for meta-model)"""
    
    df = df.copy()
    
    # Rank within each group
    df['position'] = df.groupby(group_col)[score_col].rank(
        ascending=False, method='first'
    )
    
    df['is_top3'] = (df['position'] <= 3).astype(int)
    df['is_top10'] = (df['position'] <= 10).astype(int)
    
    return df
