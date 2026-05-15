"""Folium map builder for the map screen."""
import folium


def build_folium_map(user_lat: float, user_lon: float, restaurants: list) -> folium.Map:
    fmap = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=False,
    )

    folium.CircleMarker(
        location=[user_lat, user_lon],
        radius=10,
        color="#4A7BF7",
        fill=True,
        fill_color="#4A7BF7",
        fill_opacity=1.0,
        popup="Vị trí của bạn",
    ).add_to(fmap)

    for r in restaurants:
        price_k = int(r["price"]) // 1000
        html = (
            f"<div style=\"background:#FF6B2B;color:white;padding:4px 10px;"
            f"border-radius:14px;font-size:12px;font-weight:700;white-space:nowrap;"
            f"box-shadow:0 2px 6px rgba(0,0,0,0.3);\">📍 {price_k}K</div>"
        )
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=folium.DivIcon(html=html, icon_size=(80, 28), icon_anchor=(40, 14)),
            popup=r["shop_name"],
        ).add_to(fmap)

    return fmap
