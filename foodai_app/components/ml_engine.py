"""Decision Tree taste learning."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

CUISINE_MAP     = {"vie": 0, "kor": 1, "jpn": 2, "us": 3, "tha": 4, "chn": 5, "kh": 6}
MEAL_SIZE_MAP   = {"snack": 0, "light_meal": 1, "full_meal": 2}
MEAL_FORM_MAP   = {"dry": 0, "liquid": 1, "finger": 2}
PRICE_RANGE_MAP = {"cheap": 0, "standard": 1, "premium": 2, "luxury": 3}
NUTRITION_MAP   = {"normal": 0, "high_protein": 1, "greasy": 2, "sweet": 3, "vegetarian": 4}
CALORIES_MAP    = {"low": 0, "medium": 1, "high": 2}


def encode_food(item: dict) -> list:
    return [
        CUISINE_MAP.get(item.get("cuisine", "vie"), 0),
        MEAL_SIZE_MAP.get(item.get("meal_size", "full_meal"), 1),
        MEAL_FORM_MAP.get(item.get("meal_form", "dry"), 0),
        PRICE_RANGE_MAP.get(item.get("price_range", "standard"), 1),
        NUTRITION_MAP.get(item.get("nutrition_profile", "normal"), 0),
        CALORIES_MAP.get(item.get("calories_estimate", "medium"), 1),
    ]


def generate_cold_start_data(health_target: str, food_df) -> tuple:
    """Seed training data từ health_target khi chưa có lịch sử."""
    features, labels = [], []

    SEED_RULES = {
        "diet": lambda row: 1 if row["nutrition_profile"] in ["normal", "vegetarian"]
                                  and row["calories_estimate"] == "low"
                             else (0 if row["nutrition_profile"] in ["greasy", "sweet"] else -1),
        "balanced": lambda row: 1 if row["nutrition_profile"] in ["normal", "high_protein"]
                                else (0 if row["calories_estimate"] == "high"
                                         and row["nutrition_profile"] == "greasy" else -1),
        "bulking": lambda row: 1 if row["nutrition_profile"] == "high_protein"
                                    or row["calories_estimate"] == "high"
                               else (0 if row["calories_estimate"] == "low" else -1),
    }

    rule = SEED_RULES.get(health_target, SEED_RULES["balanced"])
    for _, row in food_df.iterrows():
        label = rule(row)
        if label != -1:
            features.append(encode_food(row.to_dict()))
            labels.append(label)

    return features, labels


def train_model(features: list, labels: list):
    if len(features) < 2 or len(set(labels)) < 2:
        return None
    try:
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(features, labels)
        return clf
    except Exception:
        return None


def predict_like_proba(model, item: dict) -> float:
    if model is None:
        return 0.5
    try:
        feat = [encode_food(item)]
        proba = model.predict_proba(feat)[0]
        classes = model.classes_
        like_idx = np.where(classes == 1)[0]
        return float(proba[like_idx[0]]) if len(like_idx) > 0 else 0.5
    except Exception:
        return 0.5
