import os

import joblib
import pandas as pd
from langchain_core.tools import tool

from src.config import DATASET_PATH, ENCODER_PATH, MODEL_PATH, SCALER_PATH

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


# EXTRA TOOL
@tool
def recommend_strategic_opening(
    player_rating: int, opponent_rating: int, player_color: str
) -> str:
    """
    Queries historical match data to find the most successful chess openings
    based on the player's color and the rating difference between the two players.
    'player_color' must be either 'white' or 'black'.
    """
    try:
        # Load the dataset
        df = pd.read_csv(DATASET_PATH)

        # Normalize color input to match our dataset ('White' or 'Black')
        color = player_color.strip().lower()
        if color not in ["white", "black"]:
            return "Error: The agent must specify whether the player is playing 'white' or 'black'."

        rating_diff = player_rating - opponent_rating
        min_games = 50

        # 2. Filter the data strictly from the perspective of the user's color
        if color == "white":
            # Rating diff from White's perspective
            if rating_diff <= -100:
                target_games = df[(df["white_rating"] - df["black_rating"]) <= -100]
                scenario_name = "White Underdog (100+ point disadvantage)"
                advice = "Playing as White against a stronger opponent, focus on openings that dictate the pace but keep the position solid to avoid early tactical traps."
            elif rating_diff >= 100:
                target_games = df[(df["white_rating"] - df["black_rating"]) >= 100]
                scenario_name = "White Favorite (100+ point advantage)"
                advice = "As the stronger player with the White pieces, these openings maximize your first-move advantage to aggressively pressure weaker opponents."
            else:
                target_games = df[abs(df["white_rating"] - df["black_rating"]) < 100]
                scenario_name = "White, Evenly Matched"
                advice = "In an even match, these White openings yield the highest statistical conversion rate."

        else:
            # Rating diff from Black's perspective
            if rating_diff <= -100:
                target_games = df[(df["black_rating"] - df["white_rating"]) <= -100]
                scenario_name = "Black Underdog (100+ point disadvantage)"
                advice = "Playing Black against a stronger opponent is tough. Data suggests sharp, asymmetric defenses give the highest statistical chance of a counter-attacking upset."
            elif rating_diff >= 100:
                target_games = df[(df["black_rating"] - df["white_rating"]) >= 100]
                scenario_name = "Black Favorite (100+ point advantage)"
                advice = "As the stronger player with Black, these defenses allow you to safely equalize and outplay your opponent in the middlegame."
            else:
                target_games = df[abs(df["white_rating"] - df["black_rating"]) < 100]
                scenario_name = "Black, Evenly Matched"
                advice = "In an even match, these Black defenses provide the best winning chances without compromising solid structure."

        # 3. Find games where the user's specific color actually WON
        target_wins = target_games[target_games["winner"] == color]

        # --- CALCULATE FINAL STATISTICS ---
        win_counts = target_wins.groupby("opening_name").size()
        total_counts = target_games.groupby("opening_name").size()

        stats = pd.DataFrame({"wins": win_counts, "total_games": total_counts}).fillna(
            0
        )

        # Filter out insignificant data
        stats = stats[stats["total_games"] >= min_games]

        # Calculate the true win percentage for that specific color
        stats["win_rate"] = (stats["wins"] / stats["total_games"]) * 100

        # Sort to find the top 3
        top_3 = stats.sort_values(by="win_rate", ascending=False).head(3)

        if top_3.empty:
            return f"Not enough historical data for {color} in this specific rating gap to make a statistically confident recommendation."

        recommendations = ""
        for i, (opening, row) in enumerate(top_3.iterrows(), 1):
            recommendations += f"{i}. **{opening}** (Win Rate: {row['win_rate']:.1f}% for {color} across {int(row['total_games'])} games)\n"

        return f"### Historical Data Recommendations: {scenario_name}\n\n{recommendations}\n**Strategic Advice:** {advice}"

    except Exception as e:
        return f"Could not calculate opening statistics: {str(e)}"
