import os

import joblib
import pandas as pd
from langchain_core.tools import tool

from src.config import ENCODER_PATH, MODEL_PATH, SCALER_PATH

if (
    os.path.exists(MODEL_PATH)
    and os.path.exists(SCALER_PATH)
    and os.path.exists(ENCODER_PATH)
):
    best_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    target_encoder = joblib.load(ENCODER_PATH)
else:
    best_model = None
    scaler = None
    target_encoder = None
    print("WARNING: ML artifacts missing! Check the models/ directory.")

# Class mapping based on your API configuration
CLASS_MAPPING = {0: "Black", 1: "Draw", 2: "White"}


#
# Import custom feature engineering step from HW1
def engineer_domain_features(X_train, X_validation, X_test):
    """
    Creates new domain-knowledge features and adds them alongside
    the original columns without dropping anything.
    """
    # for the api
    datasets = [X_train]

    # for training
    if isinstance(X_validation, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        datasets = [X_train, X_validation, X_test]

    for X in datasets:
        # --- Feature 1: Rating Advantage ---
        X["rating_advantage"] = X["white_rating"] - X["black_rating"]

        # --- Feature 2: Game Duration (Minutes) ---
        # Convert the massive Unix millisecond integers to datetime objects
        start_dt = pd.to_datetime(X["created_at"], unit="ms")
        end_dt = pd.to_datetime(X["last_move_at"], unit="ms")

        # Calculate the difference and convert to total minutes
        X["game_duration_mins"] = (end_dt - start_dt).dt.total_seconds() / 60

    return X_train, X_validation, X_test


# Langgraph tool definition
@tool
def predict_chess_match_outcome(
    rated: bool,
    created_at: float,
    last_move_at: float,
    turns: int,
    white_rating: int,
    black_rating: int,
    opening_ply: int,
    increment_code: str,
    opening_eco: str,
    opening_name: str,
) -> str:
    """
    Predicts the outcome of a chess match using a trained XGBoost classifier.
    Use this tool whenever a user asks to forecast, simulate, or evaluate the outcome
    of a specific match setup based on ratings, time controls, and opening moves.

    Returns a human-readable prediction string along with confidence scores.
    """
    if best_model is None or scaler is None or target_encoder is None:
        return "Error: Prediction tool is currently unavailable. Core model artifacts are missing."

    try:
        # 1. Replicate input dictionary from the API's Pydantic model
        input_dict = {
            "rated": rated,
            "created_at": created_at,
            "last_move_at": last_move_at,
            "turns": turns,
            "white_rating": white_rating,
            "black_rating": black_rating,
            "opening_ply": opening_ply,
            "increment_code": increment_code,
            "opening_eco": opening_eco,
            "opening_name": opening_name,
        }

        # Convert dictionary to Pandas DataFrame
        input_df = pd.DataFrame([input_dict])

        # 2. Run domain feature engineering
        input_df, _, _ = engineer_domain_features(input_df, None, None)

        # 3. Categorical Preprocessing
        input_df["rated"] = input_df["rated"].astype(int)
        high_cardinality_features = ["increment_code", "opening_eco", "opening_name"]

        # Target Encoding transformation using saved parameters
        new_col_names = target_encoder.get_feature_names_out(high_cardinality_features)
        encoded_data = target_encoder.transform(input_df[high_cardinality_features])
        encoded_df = pd.DataFrame(
            encoded_data, columns=new_col_names, index=input_df.index
        )
        input_df = input_df.drop(columns=high_cardinality_features)
        input_df = pd.concat([input_df, encoded_df], axis=1)

        # 4. Feature Scaling
        cols_to_scale = [
            "turns",
            "white_rating",
            "black_rating",
            "opening_ply",
            "created_at",
            "last_move_at",
            "rating_advantage",
            "game_duration_mins",
        ]
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

        # 5. Column ordering constraint enforcement for XGBoost
        expected_columns = [
            "rated",
            "created_at",
            "last_move_at",
            "turns",
            "white_rating",
            "black_rating",
            "opening_ply",
            "rating_advantage",
            "game_duration_mins",
            "increment_code_0",
            "increment_code_1",
            "increment_code_2",
            "opening_eco_0",
            "opening_eco_1",
            "opening_eco_2",
            "opening_name_0",
            "opening_name_1",
            "opening_name_2",
        ]
        input_df = input_df[expected_columns]

        # 6. Execute Predictions & Probabilities
        prediction = best_model.predict(input_df)
        prediction_num = int(prediction[0])
        probabilities = best_model.predict_proba(input_df)[0]

        # 7. Format clean markdown output readable by the LangGraph agent
        outcome_label = CLASS_MAPPING.get(prediction_num, "Unknown")

        output_msg = (
            f"### Match Prediction Analysis\n"
            f"- **Predicted Outcome**: {outcome_label}\n"
            f"- **Model Probabilities**:\n"
            f"  - Black Win: {probabilities[0]:.2%}\n"
            f"  - Draw: {probabilities[1]:.2%}\n"
            f"  - White Win: {probabilities[2]:.2%}"
        )
        return output_msg

    except Exception as e:
        return f"Prediction Tool Error: Failed to compute prediction due to: {str(e)}"
