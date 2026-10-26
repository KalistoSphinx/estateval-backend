from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class HouseFeatures(BaseModel):
    City: str = Field(default="Bangalore", description="City location")
    Locality_Tier: str = Field(default="Mid", description="Locality tier: Budget, Mid, Premium")
    BHK: int = Field(default=2, ge=1, le=10, description="Number of bedrooms, hall, kitchen")
    Bathrooms: int = Field(default=2, ge=1, le=10, description="Number of bathrooms")
    Super_Area_sqft: float = Field(default=1050.0, ge=200, le=10000, description="Total super built-up area in sqft")
    Carpet_Area_sqft: float = Field(default=820.0, ge=150, le=8000, description="Net usable carpet area in sqft")
    Floor_No: int = Field(default=5, ge=0, le=100, description="Floor number of the apartment")
    Total_Floors: int = Field(default=14, ge=1, le=100, description="Total floors in building")
    Property_Age_years: int = Field(default=5, ge=0, le=100, description="Age of property in years")
    Parking: int = Field(default=1, ge=0, le=1, description="Dedicated parking space (1=Yes, 0=No)")
    Furnishing: str = Field(default="Semi-Furnished", description="Furnishing status: Unfurnished, Semi-Furnished, Fully-Furnished")
    Lift: int = Field(default=1, ge=0, le=1, description="Lift facility available (1=Yes, 0=No)")
    Gated_Society: int = Field(default=1, ge=0, le=1, description="Gated community (1=Yes, 0=No)")
    Distance_to_Metro_km: float = Field(default=1.8, ge=0.0, le=50.0, description="Distance to nearest metro in km")
    Distance_to_CityCenter_km: float = Field(default=12.0, ge=0.0, le=100.0, description="Distance to city center in km")
    Nearby_School_km: float = Field(default=2.5, ge=0.0, le=30.0, description="Distance to nearest school in km")
    Nearby_Hospital_km: float = Field(default=2.8, ge=0.0, le=30.0, description="Distance to nearest hospital in km")
    Crime_Rate_Index: float = Field(default=35.0, ge=0.0, le=100.0, description="Crime rate safety index score (lower is safer)")
    model: Optional[str] = Field(default="all", description="Model to use: 'all', 'linear_regression', 'knn', or 'random_forest'")

class ModelPrediction(BaseModel):
    model_key: str
    model_name: str
    predicted_price_inr: float
    price_in_lakhs: float
    price_in_crores: float
    formatted_inr: str
    price_per_sqft: float
    price_tier: str
    latency_ms: float

class MultiPredictionResponse(BaseModel):
    input_summary: Dict[str, Any]
    predictions: Dict[str, ModelPrediction]
    ensemble_average_inr: float
    ensemble_average_formatted: str
    price_spread_percentage: float
    recommended_model: str
    timestamp: str
