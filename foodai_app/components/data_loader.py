"""CSV loading, deck generation, distance + restaurant filtering."""
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .fuzzy_engine import score_food_item
from .ml_engine import encode_food, generate_cold_start_data, predict_like_proba, train_model


CUISINE_LABELS = {
    "vie": "Việt", "kor": "Hàn", "jpn": "Nhật", "us": "Âu-Mỹ",
    "tha": "Thái", "chn": "Hoa", "kh": "Campuchia",
}
PRICE_LABELS = {"cheap": "Rẻ", "standard": "Vừa", "premium": "Cao cấp", "luxury": "Sang chảnh"}
PRICE_LABEL_COLORS = {
    "cheap": "#27AE60", "standard": "#2980B9",
    "premium": "#E67E22", "luxury": "#8E44AD",
}
CUISINE_EMOJI = {
    "vie": "🍜", "kor": "🍱", "jpn": "🍣", "us": "🍔",
    "tha": "🥢", "chn": "🥟", "kh": "🍲",
}

REQUIRED_FIELDS = [
    "id", "shop_name", "food_name", "meal_size", "meal_form", "cuisine",
    "price", "price_range", "lat", "lon", "calories_estimate", "nutrition_profile",
]


def load_food_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load CSV, normalize, drop bad rows, enrich with derived columns."""
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "data" / "food_data_mockup.csv"

    df = pd.read_csv(csv_path)

    # CSV has a duplicate `food_name` header — pandas renames the second one.
    dupe_cols = [c for c in df.columns if c.startswith("food_name.") or c == "food_name.1"]
    if dupe_cols:
        df = df.drop(columns=dupe_cols)

    # Drop rows missing required fields.
    df = df.dropna(subset=REQUIRED_FIELDS)

    # Type coercion.
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["price", "lat", "lon"])
    df["price"] = df["price"].astype(int)

    # Stable mock rating per shop.
    rating_ranges = {
        "cheap": (3.8, 4.3),
        "standard": (4.0, 4.7),
        "premium": (4.3, 4.9),
        "luxury": (4.5, 5.0),
    }

    def mock_rating(row):
        lo, hi = rating_ranges.get(row["price_range"], (4.0, 4.7))
        rng = np.random.default_rng(hash(row["shop_name"]) % (2**32))
        return round(float(rng.uniform(lo, hi)), 1)

    df["rating"] = df.apply(mock_rating, axis=1)

    # Description fallback.
    if "description" not in df.columns:
        df["description"] = df["food_name"].apply(lambda n: f"{n} thơm ngon đặc trưng")
    else:
        df["description"] = df["description"].fillna(
            df["food_name"].apply(lambda n: f"{n} thơm ngon đặc trưng")
        )

    df["cuisine_label"]      = df["cuisine"].map(CUISINE_LABELS).fillna("Khác")
    df["price_label"]        = df["price_range"].map(PRICE_LABELS).fillna("Vừa")
    df["price_label_color"]  = df["price_range"].map(PRICE_LABEL_COLORS).fillna("#2980B9")
    df["cuisine_emoji"]      = df["cuisine"].map(CUISINE_EMOJI).fillna("🍽️")

    if "image_url" not in df.columns:
        df["image_url"] = ""
    df["image_url"] = df["image_url"].fillna("")

    return df.reset_index(drop=True)


# ── Haversine + restaurant filtering ──────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def estimate_district(lat: float, lon: float) -> str:
    if lat > 10.79:
        return "Bình Thạnh"
    if lat > 10.77 and lon > 106.69:
        return "Q.3"
    if lat > 10.77:
        return "Q.10"
    if lon < 106.66:
        return "Q.5"
    if lon < 106.68:
        return "Q.1"
    return "Q.3"


def filter_and_sort_restaurants(
    food_df: pd.DataFrame,
    winner_food_name: str,
    budget: int,
    user_lat: float,
    user_lon: float,
    time_available: int,
) -> list:
    mask = (food_df["food_name"] == winner_food_name) & (food_df["price"] <= budget)
    candidates = food_df[mask].copy()

    if candidates.empty:
        candidates = food_df[food_df["food_name"] == winner_food_name].copy()
    if candidates.empty:
        return []

    candidates["_distance_km"] = candidates.apply(
        lambda r: haversine(user_lat, user_lon, r["lat"], r["lon"]), axis=1
    )
    candidates["_walk_min"] = candidates["_distance_km"].apply(
        lambda d: math.ceil(d / 4.0 * 60)
    )

    radius_map = {15: 1.0, 30: 3.0, 45: 5.0, 60: 10.0}
    radius = radius_map.get(time_available, 3.0)
    in_radius = candidates[candidates["_distance_km"] <= radius]
    if in_radius.empty:
        in_radius = candidates

    if time_available <= 30:
        sorted_df = in_radius.sort_values("_distance_km")
    else:
        sorted_df = in_radius.sort_values("rating", ascending=False)

    return sorted_df.to_dict("records")


# ── Deck generation ───────────────────────────────────────────

def generate_deck(context: dict, profile: dict, ml_data: dict, food_df: pd.DataFrame, fis_sim) -> list:
    budget = context["budget"]
    filtered = food_df[food_df["price"] <= budget].copy()

    if filtered.empty:
        filtered = food_df.nsmallest(min(20, len(food_df)), "price").copy()

    features = list(ml_data.get("features", []))
    labels   = list(ml_data.get("labels", []))

    if len(features) < 4:
        seed_f, seed_l = generate_cold_start_data(profile["health_target"], filtered)
        features = seed_f + features
        labels   = seed_l + labels

    model = train_model(features, labels)

    scored_items = []
    for _, row in filtered.iterrows():
        item = row.to_dict()
        fuzzy_score = score_food_item(item, context, profile["health_target"], fis_sim)
        dt_score    = predict_like_proba(model, item)
        item["_score"] = fuzzy_score * 0.6 + dt_score * 10 * 0.4
        scored_items.append(item)

    if not scored_items:
        return []

    scored_items.sort(key=lambda x: x["_score"], reverse=True)

    if profile["ai_mode"] == "routine":
        top_items = scored_items[:10]
    else:
        top_7 = scored_items[:7]
        rest  = scored_items[7:]
        wildcards = random.sample(rest, min(3, len(rest))) if rest else []
        top_items = top_7 + wildcards

    deck = top_items[:10]
    random.shuffle(deck)
    return deck


# ── Mock walking directions ──────────────────────────────────

HCMC_STREETS = [
    "Nguyễn Huệ", "Lê Lợi", "Hai Bà Trưng", "Điện Biên Phủ",
    "Nguyễn Trãi", "Lê Văn Sỹ", "Nam Kỳ Khởi Nghĩa", "Cống Quỳnh",
    "Võ Văn Tần", "Trần Hưng Đạo",
]


def generate_mock_directions(distance_km: float, shop_name: str) -> list:
    dist_m = max(int(distance_km * 1000), 50)
    seed = hash(shop_name) % (2**32)
    rng = random.Random(seed)
    streets = rng.sample(HCMC_STREETS, 2)
    side = "trái" if seed % 2 == 0 else "phải"

    if dist_m < 300:
        return [f"Đi thẳng {dist_m}m, quán ở bên tay {side}"]
    if dist_m < 700:
        seg1 = dist_m // 2
        seg2 = dist_m - seg1
        return [
            f"Đi thẳng {seg1}m trên đường {streets[0]}",
            f"Rẽ {'phải' if side == 'trái' else 'trái'} vào đường {streets[1]}, đi {seg2}m",
        ]
    seg1 = dist_m // 3
    seg2 = dist_m // 3
    seg3 = dist_m - seg1 - seg2
    return [
        f"Đi thẳng {seg1}m trên đường {streets[0]}",
        f"Rẽ phải vào đường {streets[1]}",
        f"Đi tiếp {seg3}m, quán ở bên tay {side}",
    ]
