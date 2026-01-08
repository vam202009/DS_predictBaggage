from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd

# ---------- CONFIG & MODEL LOAD ---------- #

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bag_propensity_logreg_pipeline.joblib"

# Feature lists – must match training notebook
feature_cols_num = [
    "party_size",
    "adt_count",
    "chd_count",
    "inf_count",
    "booking_horizon_days",
    "base_fare_total",
]

feature_cols_cat = [
    "traveler_type",
    "od",
    "trip_type",
    "flight_no",
    "cabin",
    "fare_family",
    "pos_country",
    "sales_channel",
    "device",
    "loyalty_tier",
    "corporate_flag",
    "dep_month",
    "dep_dow",
    "dep_season",
]

all_features = feature_cols_num + feature_cols_cat

# Load the trained pipeline
clf = joblib.load(MODEL_PATH)


def bag_target_band(p: float) -> str:
    """Simple business banding rule."""
    if p >= 0.7:
        return "HIGH"
    elif p >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


# ---------- FASTAPI APP ---------- #

app = FastAPI(
    title="Extra Baggage Propensity API",
    version="1.0.0",
    description="Predict probability that a booking will buy extra baggage."
)

# CORS (for later when you add frontend)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- REQUEST / RESPONSE SCHEMAS ---------- #

class BagRequest(BaseModel):
    # numeric
    party_size: int
    adt_count: int
    chd_count: int
    inf_count: int
    booking_horizon_days: int
    base_fare_total: float

    # categorical
    traveler_type: str
    od: str
    trip_type: str
    flight_no: str
    cabin: str
    fare_family: str
    pos_country: str
    sales_channel: str
    device: str
    loyalty_tier: str
    corporate_flag: int
    dep_month: int
    dep_dow: int
    dep_season: str


class BagResponse(BaseModel):
    probability_bag: float
    band: str


# ---------- ENDPOINTS ---------- #

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_bag", response_model=BagResponse)
def predict_bag(req: BagRequest):
    # Convert request to DataFrame
    data = req.dict()
    df_input = pd.DataFrame([data])[all_features]

    # Predict
    proba = float(clf.predict_proba(df_input)[:, 1][0])
    band = bag_target_band(proba)

    return BagResponse(probability_bag=proba, band=band)