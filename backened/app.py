
import os
import joblib
import numpy as np
import pandas as pd
import mysql.connector

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from database import get_connection


# ==========================================
# 1. INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="CropCare API",
    description="Crop Stress Detection and Irrigation Priority Decision Support System",
    version="1.1"
)


# ==========================================
# 2. LOAD TRAINED MODELS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_stress_models.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found: " + MODEL_PATH)

bundle = joblib.load(MODEL_PATH)

classifier = bundle["classification_pipeline"]
regressor = bundle["regression_pipeline"]
feature_cols = bundle["feature_columns"]

print("Random Forest models loaded successfully!")


# ==========================================
# 3. REQUEST SCHEMAS
# ==========================================

class CropInput(BaseModel):
    features: Dict[str, Any]


class FieldCreate(BaseModel):
    farmer_id: int
    field_name: str
    area_acres: float
    crop_type: str
    soil_type: str
    irrigation_method: str
    growth_stage: str


class ObservationInput(BaseModel):
    soil_moisture_m3_m3: float
    soil_temperature_c: float
    air_temperature_c: float
    relative_humidity_pct: float
    rainfall_mm_day: float
    et0_mm_day: float
    actual_et_mm_day: float
    ndvi: float
    soil_ph: float
    soil_organic_carbon_pct: float
    nitrogen_kg_ha: float
    phosphorus_kg_ha: float
    potassium_kg_ha: float
    electrical_conductivity_ds_m: float
    irrigated_in_last_7_days: bool


# ==========================================
# 4. IRRIGATION PRIORITY CALCULATION
# ==========================================

def calculate_irrigation_priority(data, stress_index):

    moisture_deficit = np.clip(
        (0.27 - data["soil_moisture_m3_m3"].to_numpy()) / 0.22,
        0, 1
    )

    rainfall_relief = np.clip(
        data["rainfall_mm_day"].to_numpy() / 18,
        0, 1
    )

    score = np.clip(
        100 * (
            0.72 * np.clip(stress_index, 0, 1)
            + 0.20 * moisture_deficit
            + 0.08 * (1 - rainfall_relief)
        ),
        0, 100
    )

    category = pd.cut(
        score,
        bins=[-0.001, 25, 50, 75, 100.001],
        labels=[
            "Low Priority",
            "Moderate Priority",
            "High Priority",
            "Critical Priority"
        ]
    )

    return float(score[0]), str(category[0])


# ==========================================
# 5. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Welcome to CropCare API",
        "status": "success"
    }


# ==========================================
# 6. HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():
    try:
        connection = get_connection()
        connection.close()

        return {
            "status": "healthy",
            "database": "connected",
            "classifier": "Random Forest",
            "regressor": "Random Forest"
        }

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {str(e)}"
        )


# ==========================================
# 7. ORIGINAL PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict_crop_stress(input_data: CropInput):

    features = input_data.features

    missing = [
        col for col in feature_cols
        if col not in features
    ]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing}
        )

    try:
        input_df = pd.DataFrame([
            {col: features[col] for col in feature_cols}
        ])

        predicted_risk = classifier.predict(input_df)[0]

        predicted_stress = float(np.clip(
            regressor.predict(input_df)[0],
            0, 1
        ))

        priority_score, priority_category = (
            calculate_irrigation_priority(
                input_df,
                np.array([predicted_stress])
            )
        )

        return {
            "status": "success",
            "predicted_stress_risk": str(predicted_risk),
            "predicted_stress_index": round(predicted_stress, 3),
            "irrigation_priority_score": round(priority_score, 1),
            "irrigation_priority_category": priority_category
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# 8. AREAS API
# ==========================================

@app.get("/areas")
def get_areas():

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                s.state_id,
                s.state_name,
                d.district_id,
                d.district_name,
                m.mandal_id,
                m.mandal_name,
                v.village_id,
                v.village_name
            FROM states s
            LEFT JOIN districts d ON s.state_id = d.state_id
            LEFT JOIN mandals m ON d.district_id = m.district_id
            LEFT JOIN villages v ON m.mandal_id = v.mandal_id
            ORDER BY s.state_name, d.district_name,
                     m.mandal_name, v.village_name
        """)

        areas = cursor.fetchall()

        return {
            "status": "success",
            "total_records": len(areas),
            "areas": areas
        }

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ==========================================
# 9. FARMERS API
# ==========================================

@app.get("/farmers")
def get_farmers():

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                f.farmer_id,
                f.farmer_name,
                f.phone,
                v.village_name,
                m.mandal_name,
                d.district_name,
                s.state_name,
                COUNT(fl.field_id) AS total_fields
            FROM farmers f
            JOIN villages v ON f.village_id = v.village_id
            JOIN mandals m ON v.mandal_id = m.mandal_id
            JOIN districts d ON m.district_id = d.district_id
            JOIN states s ON d.state_id = s.state_id
            LEFT JOIN fields fl ON f.farmer_id = fl.farmer_id
            GROUP BY
                f.farmer_id, f.farmer_name, f.phone,
                v.village_name, m.mandal_name,
                d.district_name, s.state_name
            ORDER BY f.farmer_name
        """)

        farmers = cursor.fetchall()

        return {
            "status": "success",
            "total_farmers": len(farmers),
            "farmers": farmers
        }

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ==========================================
# 10. GET FIELDS OF A FARMER
# ==========================================

@app.get("/farmers/{farmer_id}/fields")
def get_farmer_fields(farmer_id: int):

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                field_id,
                farmer_id,
                field_name,
                area_acres,
                crop_type,
                soil_type,
                irrigation_method,
                growth_stage
            FROM fields
            WHERE farmer_id = %s
            ORDER BY field_id DESC
        """, (farmer_id,))

        fields = cursor.fetchall()

        return {
            "status": "success",
            "farmer_id": farmer_id,
            "total_fields": len(fields),
            "fields": fields
        }

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ==========================================
# 11. ADD A NEW FIELD
# ==========================================

@app.post("/fields")
def create_field(field: FieldCreate):

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor()

        # Check whether farmer exists
        cursor.execute(
            "SELECT farmer_id FROM farmers WHERE farmer_id = %s",
            (field.farmer_id,)
        )

        if not cursor.fetchone():
            raise HTTPException(
                status_code=404,
                detail="Farmer not found"
            )

        cursor.execute("""
            INSERT INTO fields (
                farmer_id,
                field_name,
                area_acres,
                crop_type,
                soil_type,
                irrigation_method,
                growth_stage
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            field.farmer_id,
            field.field_name,
            field.area_acres,
            field.crop_type,
            field.soil_type,
            field.irrigation_method,
            field.growth_stage
        ))

        connection.commit()

        return {
            "status": "success",
            "message": "Field created successfully",
            "field_id": cursor.lastrowid
        }

    except HTTPException:
        raise

    except mysql.connector.Error as e:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# ==========================================
# 12. ANALYZE FIELD AND SAVE REPORT
# ==========================================

@app.post("/analyze/{field_id}")
def analyze_field(field_id: int, observation: ObservationInput):

    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        # Get field and its location information
        cursor.execute("""
            SELECT
                s.state_name,
                d.district_name,
                fl.crop_type,
                fl.soil_type,
                fl.irrigation_method,
                fl.growth_stage
            FROM fields fl
            JOIN farmers f ON fl.farmer_id = f.farmer_id
            JOIN villages v ON f.village_id = v.village_id
            JOIN mandals m ON v.mandal_id = m.mandal_id
            JOIN districts d ON m.district_id = d.district_id
            JOIN states s ON d.state_id = s.state_id
            WHERE fl.field_id = %s
        """, (field_id,))

        field_data = cursor.fetchone()

        if not field_data:
            raise HTTPException(
                status_code=404,
                detail="Field not found"
            )

        # Combine field information and observation data
        features = {
            **field_data,
            **observation.model_dump()
        }

        # Ensure feature names match the trained model
        missing = [
            col for col in feature_cols
            if col not in features
        ]

        if missing:
            raise HTTPException(
                status_code=422,
                detail={"missing_features": missing}
            )

        input_df = pd.DataFrame([
            {col: features[col] for col in feature_cols}
        ])

        # Classification
        predicted_risk = classifier.predict(input_df)[0]

        # Regression
        predicted_stress = float(np.clip(
            regressor.predict(input_df)[0],
            0, 1
        ))

        # Irrigation priority
        priority_score, priority_category = (
            calculate_irrigation_priority(
                input_df,
                np.array([predicted_stress])
            )
        )

        # Save observation
        cursor.execute("""
            INSERT INTO field_observations (
                field_id,
                observation_date,
                soil_moisture_m3_m3,
                soil_temperature_c,
                air_temperature_c,
                relative_humidity_pct,
                rainfall_mm_day,
                et0_mm_day,
                actual_et_mm_day,
                ndvi,
                soil_ph,
                soil_organic_carbon_pct,
                nitrogen_kg_ha,
                phosphorus_kg_ha,
                potassium_kg_ha,
                electrical_conductivity_ds_m,
                irrigated_in_last_7_days
            )
            VALUES (
                %s, CURDATE(), %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """, (
            field_id,
            observation.soil_moisture_m3_m3,
            observation.soil_temperature_c,
            observation.air_temperature_c,
            observation.relative_humidity_pct,
            observation.rainfall_mm_day,
            observation.et0_mm_day,
            observation.actual_et_mm_day,
            observation.ndvi,
            observation.soil_ph,
            observation.soil_organic_carbon_pct,
            observation.nitrogen_kg_ha,
            observation.phosphorus_kg_ha,
            observation.potassium_kg_ha,
            observation.electrical_conductivity_ds_m,
            observation.irrigated_in_last_7_days
        ))

        observation_id = cursor.lastrowid

        # Save analysis report
        cursor.execute("""
            INSERT INTO analysis_reports (
                field_id,
                observation_id,
                predicted_stress_risk,
                predicted_stress_index,
                irrigation_priority_score,
                irrigation_priority_category
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            field_id,
            observation_id,
            str(predicted_risk),
            predicted_stress,
            priority_score,
            priority_category
        ))

        report_id = cursor.lastrowid

        connection.commit()

        return {
            "status": "success",
            "message": "Field analyzed and report saved successfully",
            "field_id": field_id,
            "observation_id": observation_id,
            "report_id": report_id,
            "predicted_stress_risk": str(predicted_risk),
            "predicted_stress_index": round(predicted_stress, 3),
            "irrigation_priority_score": round(priority_score, 1),
            "irrigation_priority_category": priority_category
        }

    except HTTPException:
        if connection:
            connection.rollback()
        raise

    except Exception as e:
        if connection:
            connection.rollback()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
