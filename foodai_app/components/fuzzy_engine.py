"""Fuzzy Inference System (Mamdani) for food suitability scoring."""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


HEALTH_MULTIPLIERS = {
    "diet": {
        "normal": 1.0, "high_protein": 1.1, "greasy": 0.5,
        "sweet": 0.5, "vegetarian": 1.2,
    },
    "balanced": {
        "normal": 1.0, "high_protein": 1.05, "greasy": 0.85,
        "sweet": 0.85, "vegetarian": 1.0,
    },
    "bulking": {
        "normal": 1.0, "high_protein": 1.3, "greasy": 1.0,
        "sweet": 0.8, "vegetarian": 0.9,
    },
}


def build_fuzzy_system():
    """Build & cache the FIS once. Returns a ControlSystemSimulation."""
    price_fit = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "price_fit")
    price_fit["poor"] = fuzz.trapmf(price_fit.universe, [0, 0, 0.2, 0.4])
    price_fit["fair"] = fuzz.trimf(price_fit.universe, [0.3, 0.55, 0.75])
    price_fit["good"] = fuzz.trapmf(price_fit.universe, [0.65, 0.85, 1.0, 1.0])

    hunger_match = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "hunger_match")
    hunger_match["low"]    = fuzz.trapmf(hunger_match.universe, [0, 0, 2, 4])
    hunger_match["medium"] = fuzz.trimf(hunger_match.universe, [3, 5, 7])
    hunger_match["high"]   = fuzz.trapmf(hunger_match.universe, [6, 8, 10, 10])

    time_match = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "time_match")
    time_match["low"]    = fuzz.trapmf(time_match.universe, [0, 0, 2, 4])
    time_match["medium"] = fuzz.trimf(time_match.universe, [3, 5, 7])
    time_match["high"]   = fuzz.trapmf(time_match.universe, [6, 8, 10, 10])

    weather_match = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "weather_match")
    weather_match["low"]    = fuzz.trapmf(weather_match.universe, [0, 0, 2, 4])
    weather_match["medium"] = fuzz.trimf(weather_match.universe, [3, 5, 7])
    weather_match["high"]   = fuzz.trapmf(weather_match.universe, [6, 8, 10, 10])

    suitability = ctrl.Consequent(np.arange(0, 10.1, 0.1), "suitability")
    suitability["very_low"]  = fuzz.trapmf(suitability.universe, [0, 0, 1.5, 3])
    suitability["low"]       = fuzz.trimf(suitability.universe, [2, 3.5, 5])
    suitability["medium"]    = fuzz.trimf(suitability.universe, [4, 5.5, 7])
    suitability["high"]      = fuzz.trimf(suitability.universe, [6, 7.5, 9])
    suitability["very_high"] = fuzz.trapmf(suitability.universe, [8, 9, 10, 10])

    rules = [
        ctrl.Rule(price_fit["poor"], suitability["very_low"]),
        ctrl.Rule(price_fit["good"] & hunger_match["high"] & time_match["high"], suitability["very_high"]),
        ctrl.Rule(price_fit["good"] & hunger_match["high"] & time_match["medium"], suitability["high"]),
        ctrl.Rule(price_fit["good"] & hunger_match["medium"] & time_match["high"], suitability["high"]),
        ctrl.Rule(weather_match["high"] & price_fit["good"], suitability["high"]),
        ctrl.Rule(weather_match["high"] & hunger_match["high"], suitability["high"]),
        ctrl.Rule(weather_match["low"] & hunger_match["high"], suitability["medium"]),
        ctrl.Rule(price_fit["fair"] & hunger_match["high"], suitability["medium"]),
        ctrl.Rule(price_fit["fair"] & hunger_match["medium"] & time_match["medium"], suitability["medium"]),
        ctrl.Rule(hunger_match["low"], suitability["low"]),
        ctrl.Rule(time_match["low"] & price_fit["fair"], suitability["low"]),
        ctrl.Rule(time_match["low"] & price_fit["good"], suitability["medium"]),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def compute_price_fit(item_price: int, budget: int) -> float:
    if budget == 0:
        return 0.0
    if item_price <= budget:
        return min(1.0, 1.0 - (budget - item_price) / budget * 0.3)
    overage = (item_price - budget) / budget
    return max(0.0, 1.0 - overage * 2)


def compute_hunger_match(hunger_level: int, meal_size: str) -> float:
    matrix = {
        1: {"snack": 9.0, "light_meal": 6.0, "full_meal": 2.0},
        2: {"snack": 7.0, "light_meal": 9.0, "full_meal": 5.0},
        3: {"snack": 5.0, "light_meal": 7.0, "full_meal": 9.0},
        4: {"snack": 2.0, "light_meal": 5.0, "full_meal": 9.5},
        5: {"snack": 0.5, "light_meal": 3.0, "full_meal": 10.0},
    }
    return matrix.get(hunger_level, {}).get(meal_size, 5.0)


def compute_time_match(time_available: int, meal_form: str, meal_size: str) -> float:
    quick_forms = {"finger": 10, "dry": 7}
    slow_forms  = {"liquid": 5}
    base = quick_forms.get(meal_form, slow_forms.get(meal_form, 6))

    if time_available == 15:
        if meal_size == "full_meal" and meal_form == "liquid":
            return 2.0
        if meal_size == "snack":
            return min(10.0, base * 1.2)
        return base * 0.8
    if time_available == 30:
        return base
    if time_available == 45:
        return min(10.0, base * 1.1)
    return min(10.0, base * 1.2)


def compute_weather_match(weather_condition: str, meal_form: str, calories: str) -> float:
    if weather_condition == "hot":
        if meal_form == "liquid":
            return 3.0
        if calories == "low":
            return 9.0
        if calories == "medium":
            return 7.0
        return 6.0
    if weather_condition == "rainy":
        if meal_form == "liquid":
            return 9.0
        if meal_form == "finger":
            return 5.0
        return 7.0
    if weather_condition == "cold":
        if meal_form == "liquid":
            return 10.0
        if calories == "high":
            return 8.0
        return 6.0
    return 7.5


def apply_health_multiplier(score: float, nutrition_profile: str, health_target: str) -> float:
    mult = HEALTH_MULTIPLIERS.get(health_target, {}).get(nutrition_profile, 1.0)
    return min(10.0, score * mult)


def score_food_item(item: dict, context: dict, health_target: str, fis_sim) -> float:
    try:
        pf = compute_price_fit(item["price"], context["budget"])
        hm = compute_hunger_match(context["hunger_level"], item["meal_size"])
        tm = compute_time_match(context["time_available"], item["meal_form"], item["meal_size"])
        wm = compute_weather_match(
            context["weather"]["condition"], item["meal_form"], item["calories_estimate"]
        )

        fis_sim.input["price_fit"]     = float(np.clip(pf, 0, 1))
        fis_sim.input["hunger_match"]  = float(np.clip(hm, 0, 10))
        fis_sim.input["time_match"]    = float(np.clip(tm, 0, 10))
        fis_sim.input["weather_match"] = float(np.clip(wm, 0, 10))
        fis_sim.compute()

        raw_score = fis_sim.output["suitability"]
        return apply_health_multiplier(raw_score, item["nutrition_profile"], health_target)
    except Exception:
        return 5.0
