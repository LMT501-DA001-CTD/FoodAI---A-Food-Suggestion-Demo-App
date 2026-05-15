"""Open-Meteo weather integration. Silent fallback on error."""
import requests

HCMC_LAT, HCMC_LON = 10.7769, 106.7009

WEATHERCODE_MAP = {
    range(0, 3):   ("hot",    "Nắng đẹp",   "☀️"),
    range(3, 4):   ("normal", "Nhiều mây",  "⛅"),
    range(45, 78): ("rainy",  "Có mưa",     "🌧️"),
    range(80, 83): ("rainy",  "Mưa rào",    "⛈️"),
}

FALLBACK_WEATHER = {
    "temp": 32,
    "condition": "hot",
    "description": "Nắng đẹp",
    "icon": "☀️",
}


def fetch_weather(lat: float = HCMC_LAT, lon: float = HCMC_LON) -> dict:
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&timezone=Asia/Ho_Chi_Minh"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        cw = data.get("current_weather", {})
        temp = cw.get("temperature", 32)
        code = int(cw.get("weathercode", 0))

        condition, description, icon = "normal", "Bình thường", "🌤️"
        for code_range, (cond, desc, ic) in WEATHERCODE_MAP.items():
            if code in code_range:
                condition, description, icon = cond, desc, ic
                break

        if temp > 32:
            condition = "hot"
        elif temp < 20:
            condition = "cold"
            description = "Trời mát"
            icon = "🌬️"

        return {
            "temp": round(temp, 1),
            "condition": condition,
            "description": description,
            "icon": icon,
        }
    except Exception:
        return dict(FALLBACK_WEATHER)
