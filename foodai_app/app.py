"""FoodAI — Smart Food Recommendation App.

Entry point. Run: uv run streamlit run app.py
"""
from __future__ import annotations

import copy
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from components.data_loader import (
    estimate_district,
    filter_and_sort_restaurants,
    generate_deck,
    generate_mock_directions,
    haversine,
    load_food_data,
)
from components.fuzzy_engine import build_fuzzy_system
from components.map_renderer import build_folium_map
from components.ml_engine import encode_food
from components.weather_api import fetch_weather

# ─── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="FoodAI 🍜",
    page_icon="🍜",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUISINE_EMOJI = {
    "vie": "🍜", "kor": "🍱", "jpn": "🍣", "us": "🍔",
    "tha": "🥢", "chn": "🥟", "kh": "🍲",
}

DEFAULT_STATE = {
    "screen": "main",
    "profile": {
        "name": "Nguyễn Văn An",
        "health_target": "balanced",
        "ai_mode": "routine",
        "location_name": "Quận 1, TP.HCM",
        "lat": 10.760898,
        "lon": 106.667891,
        "total_orders": 24,
        "total_likes": 8,
        "avg_rating": 4.8,
    },
    "context": {
        "budget": 150000,
        "hunger_level": 3,
        "time_available": 30,
        "weather": {
            "temp": 33,
            "condition": "hot",
            "description": "Nắng đẹp",
            "icon": "☀️",
        },
    },
    "deck": [],
    "deck_index": 0,
    "liked": [],
    "disliked": [],
    "super_liked": [],
    "recycled_deck": [],
    "is_recycle_round": False,
    "winner": None,
    "selected_shop": None,
    "show_route_popup": False,
    "ml_features": [],
    "ml_labels": [],
    "history": [],
    "rating_value": 0,
    "rating_tags": [],
    "info_msg": "",
}


# ─── CSS ────────────────────────────────────────────────────────
def load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<style>.main .block-container{max-width:430px!important;margin:0 auto!important;}"
            "body{background:#f5f6fc!important;}</style>",
            unsafe_allow_html=True,
        )


def init_session_state() -> None:
    for key, val in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(val)


# ─── Helpers ────────────────────────────────────────────────────
def fmt_price(value: int) -> str:
    if value < 1000:
        return f"{value} VNĐ"
    return f"{value // 1000}K VNĐ"


def goto(screen: str) -> None:
    st.session_state["screen"] = screen
    st.rerun()


def _food_image_html(item: dict, height: int = 380, font_size: int = 100) -> str:
    """Image with CSS-based emoji fallback (no JS — survives Streamlit sanitizer).

    The emoji sits behind the img; if the img loads it fully covers the emoji,
    if it 404s the emoji remains visible.
    """
    emoji = item.get("cuisine_emoji") or CUISINE_EMOJI.get(item.get("cuisine"), "🍽️")
    img_url = (item.get("image_url") or "").strip()
    img_tag = (
        f'<img src="{img_url}" alt="" referrerpolicy="no-referrer" loading="lazy" '
        f'style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;">'
        if img_url else ""
    )
    return (
        f'<div style="position:relative;width:100%;height:{height}px;'
        f'background:linear-gradient(135deg,#2c2454 0%,#FF6B2B 100%);'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-size:{font_size}px;color:rgba(255,255,255,0.85);overflow:hidden;">'
        f"{emoji}{img_tag}</div>"
    )


def pill_row(label_key: str, options: list[tuple[str, object]], state_path: tuple, cols_per_row: int):
    """Render pill-button row. options = [(label, value), ...]. state_path = (top_key, sub_key)."""
    top, sub = state_path
    current = st.session_state[top][sub]
    rows = [options[i : i + cols_per_row] for i in range(0, len(options), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (lbl, val) in zip(cols, row):
            is_selected = current == val
            btn_type = "primary" if is_selected else "secondary"
            if col.button(lbl, key=f"{label_key}-{val}", type=btn_type, use_container_width=True):
                st.session_state[top][sub] = val
                st.rerun()


# ─── Screen: MAIN ───────────────────────────────────────────────
def render_main() -> None:
    if st.session_state.get("info_msg"):
        st.info(st.session_state["info_msg"])
        st.session_state["info_msg"] = ""

    st.markdown(
        "<div style='color:#FF6B2B;font-size:0.8rem;font-weight:700;letter-spacing:2px;"
        "text-transform:uppercase;'>🍜 FOODAI • GỢI Ý MÓN ĂN</div>",
        unsafe_allow_html=True,
    )
    st.markdown("# Hôm nay bạn muốn ăn gì? 🍜")

    if st.button("👤  Gu của bạn", use_container_width=True, key="btn-profile"):
        goto("profile")

    ctx = st.session_state["context"]

    col_a, col_b = st.columns([3, 1])
    col_a.markdown("**💰 Ngân sách**")
    col_b.markdown(
        f"<div style='text-align:right;color:#FF6B2B;font-weight:700;'>{fmt_price(ctx['budget'])}</div>",
        unsafe_allow_html=True,
    )
    ctx["budget"] = st.slider(
        "budget", min_value=0, max_value=500000, value=ctx["budget"], step=5000,
        label_visibility="collapsed", format="%d",
    )
    st.caption("AI sẽ chỉ gợi ý món phù hợp trong mức ngân sách này")

    st.markdown("**😋 Bạn đang đói cỡ nào?**")
    pill_row(
        "hunger",
        [
            ("Chưa đói", 1),
            ("Đói lưng lửng", 2),
            ("Đói sương sương", 3),
            ("Đói cồn cào", 4),
            ("Đói xỉu!", 5),
        ],
        ("context", "hunger_level"),
        cols_per_row=3,
    )

    st.markdown("**⏱️ Thời gian bạn có?**")
    pill_row(
        "time",
        [("15 phút", 15), ("30 phút", 30), ("45 phút", 45), ("1 tiếng", 60)],
        ("context", "time_available"),
        cols_per_row=2,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🎯  Tìm Món Ăn", type="primary", use_container_width=True, key="cta-find"):
        with st.spinner("Đang tìm món ăn phù hợp..."):
            profile = st.session_state["profile"]
            st.session_state["context"]["weather"] = fetch_weather(profile["lat"], profile["lon"])

            if st.session_state["is_recycle_round"]:
                deck = list(st.session_state["recycled_deck"])
                st.session_state["is_recycle_round"] = False
                st.session_state["recycled_deck"] = []
            else:
                deck = generate_deck(
                    st.session_state["context"],
                    profile,
                    {
                        "features": st.session_state["ml_features"],
                        "labels": st.session_state["ml_labels"],
                    },
                    food_df,
                    fis_sim,
                )

            st.session_state["deck"] = deck
            st.session_state["deck_index"] = 0
            st.session_state["liked"] = []
            st.session_state["disliked"] = []
            st.session_state["super_liked"] = []
            st.session_state["winner"] = None

        if not deck:
            st.warning("Không tìm được món nào trong ngân sách. Thử tăng ngân sách nhé!")
            return
        goto("swipe")


# ─── Screen: PROFILE ────────────────────────────────────────────
def render_profile() -> None:
    profile = st.session_state["profile"]

    top_l, _, top_r = st.columns([1, 2, 1])
    if top_l.button("← Quay lại", key="profile-back"):
        goto("main")
    top_r.button("⚙️", disabled=True, key="profile-settings")

    st.markdown(
        f"""
        <div style="background:#FF6B2B;border-radius:16px;padding:18px;text-align:center;
                    margin:12px 0;color:white;">
          <div style="font-size:42px;">👤</div>
          <div style="font-weight:700;font-size:1.2rem;">{profile['name']}</div>
          <div style="opacity:0.9;">📍 {profile['location_name']}</div>
          <div style="margin-top:6px;background:rgba(255,255,255,0.18);display:inline-block;
                      padding:3px 10px;border-radius:12px;font-size:13px;">⭐ Thành viên Vàng</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Lần đặt", profile["total_orders"])
    c2.metric("Yêu thích", profile["total_likes"])
    c3.metric("Đánh giá", profile["avg_rating"])

    st.markdown("### 🍜 Món ngon - chuẩn gu")

    st.markdown("**Mục tiêu sức khỏe**")
    pill_row(
        "health",
        [("Giảm cân", "diet"), ("Cân bằng", "balanced"), ("Tăng cơ", "bulking")],
        ("profile", "health_target"),
        cols_per_row=3,
    )

    st.markdown("**Ưu tiên chọn món**")
    pill_row(
        "aimode",
        [("Thường lệ", "routine"), ("Khám phá", "explore")],
        ("profile", "ai_mode"),
        cols_per_row=2,
    )

    st.markdown("### 📋 Lịch sử gần đây")
    hist = st.session_state["history"]
    if not hist:
        st.info("Chưa có lịch sử. Hãy tìm món ăn đầu tiên!")
    else:
        for h in hist[:5][::-1]:
            stars = "⭐" * h["rating"] + "☆" * (5 - h["rating"])
            st.markdown(
                f"""
                <div style="display:flex;gap:10px;padding:10px;border:1px solid #2C3555;
                            border-radius:10px;margin:6px 0;background:#1E2745;">
                  <div style="font-size:30px;">{h.get('emoji','🍽️')}</div>
                  <div style="flex:1;">
                    <div style="color:white;font-weight:700;">{h['food_name']}</div>
                    <div style="color:#8A93B0;font-size:13px;">{h['shop_name']}</div>
                    <div style="color:#8A93B0;font-size:12px;">{h['timestamp']} · {fmt_price(h['price'])}</div>
                  </div>
                  <div style="color:#FFD700;">{stars}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── Screen: SWIPE ──────────────────────────────────────────────
def _check_deck_end() -> None:
    """Called after each swipe. Decides whether deck is done."""
    if st.session_state["deck_index"] < len(st.session_state["deck"]):
        return
    liked = st.session_state["liked"]
    if len(liked) == 0:
        st.session_state["info_msg"] = (
            "Không tìm được món ưng ý? Thử điều chỉnh lại ngân sách hoặc độ đói nhé!"
        )
        goto("main")
    elif len(liked) == 1:
        st.session_state["recycled_deck"] = list(st.session_state["disliked"])
        st.session_state["is_recycle_round"] = True
        st.session_state["info_msg"] = "Còn 1 món trong giỏ. Tìm thêm để so sánh nhé!"
        goto("main")


def _record_swipe(label: int, current: dict) -> None:
    st.session_state["ml_features"].append(encode_food(current))
    st.session_state["ml_labels"].append(label)
    st.session_state["deck_index"] += 1


def handle_dislike(current: dict) -> None:
    st.session_state["disliked"].append(current)
    _record_swipe(0, current)
    _check_deck_end()


def handle_like(current: dict) -> None:
    st.session_state["liked"].append(current)
    _record_swipe(1, current)
    if len(st.session_state["liked"]) >= 2:
        goto("battle")
    _check_deck_end()


def handle_super_like(current: dict) -> None:
    st.session_state["super_liked"].append(current)
    st.session_state["liked"].append(current)
    _record_swipe(1, current)
    if len(st.session_state["liked"]) >= 2:
        goto("battle")
    _check_deck_end()


def render_swipe() -> None:
    deck = st.session_state["deck"]
    idx = st.session_state["deck_index"]
    if not deck or idx >= len(deck):
        _check_deck_end()
        return

    current = deck[idx]
    profile = st.session_state["profile"]
    ctx = st.session_state["context"]

    # Top bar
    tc1, tc2, tc3 = st.columns([1, 2, 1])
    if tc1.button("← Quay lại", key="swipe-back"):
        goto("main")
    tc2.markdown(
        f"<div style='text-align:center;color:white;font-weight:600;'>"
        f"{idx + 1}/{len(deck)} &nbsp;🛒 <span style='color:#FF6B2B;'>"
        f"{len(st.session_state['liked'])}</span></div>",
        unsafe_allow_html=True,
    )
    if tc3.button("👤", key="swipe-profile"):
        goto("profile")

    # Context strip
    now = datetime.now().strftime("%H:%M")
    weather = ctx["weather"]
    st.markdown(
        f"""
        <div style="background:#1E2745;border-radius:10px;padding:8px 12px;margin:8px 0;
                    color:#cfd6f0;font-size:13px;display:flex;justify-content:space-between;">
          <div>⏰ {now}</div>
          <div>{weather['icon']} {weather['temp']}°C · {weather['description']}</div>
          <div>📍 {profile['location_name']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    in_budget = sum(1 for d in deck if d["price"] <= ctx["budget"])
    st.markdown(
        f"<div style='color:#8A93B0;font-size:13px;margin-bottom:6px;'>"
        f"Ngân sách: <b style='color:#FF6B2B;'>{fmt_price(ctx['budget'])}</b> • "
        f"{in_budget} món phù hợp</div>",
        unsafe_allow_html=True,
    )

    # Food card — image with CSS fallback, tag, and overlay
    st.markdown(
        f"""
        <div style="position:relative;border-radius:16px;overflow:hidden;width:100%;
                    background:#1E2745;margin:10px 0;">
          {_food_image_html(current, height=380, font_size=110)}
          <span style="position:absolute;top:12px;left:12px;background:#FF6B2B;color:white;
                       border-radius:12px;padding:3px 10px;font-size:12px;font-weight:600;
                       box-shadow:0 2px 6px rgba(0,0,0,0.35);">
            {current['cuisine_label']}
          </span>
          <div style="position:absolute;bottom:0;left:0;right:0;
                      background:linear-gradient(transparent,rgba(0,0,0,0.88));padding:20px;color:white;">
            <h2 style="margin:0;color:white;">{current['food_name']}</h2>
            <p style="margin:4px 0;color:#e1e6f5;font-size:13px;">{current['description']}</p>
            <p style="margin:0;color:#FF6B2B;font-weight:700;">từ {current['price']:,} VNĐ</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bc1, bc2 = st.columns(2)
    if bc1.button("❌", key=f"dislike-{idx}", use_container_width=True):
        handle_dislike(current)
        st.rerun()
    if bc2.button("❤️", key=f"like-{idx}", use_container_width=True):
        handle_like(current)
        st.rerun()


# ─── Screen: BATTLE ─────────────────────────────────────────────
def render_battle() -> None:
    liked = st.session_state["liked"]
    super_names = {s["id"] for s in st.session_state["super_liked"]}

    if len(liked) < 2:
        goto("swipe")

    if st.button("← Quay lại", key="battle-back"):
        goto("main")
    st.markdown("## Chọn 1 món nhé! 🤔")
    st.caption("Giỏ hàng đã đầy. Chọn món yêu thích để tiếp tục.")

    contenders = liked[-2:]

    cols = st.columns(2)
    for col, food in zip(cols, contenders):
        is_super = food["id"] in super_names
        border = "border:2px solid gold;" if is_super else "border:1px solid #2C3555;"
        super_badge = (
            "<div style='position:absolute;top:8px;right:8px;background:gold;color:#222;"
            "border-radius:10px;padding:2px 8px;font-size:11px;font-weight:700;'>⭐ Siêu thích</div>"
            if is_super else ""
        )

        with col:
            st.markdown(
                f"""
                <div style="position:relative;border-radius:16px;overflow:hidden;background:#1E2745;
                            margin:6px 0;{border}">
                  {_food_image_html(food, height=220, font_size=70)}
                  <span style="position:absolute;top:10px;left:10px;background:#FF6B2B;color:white;
                               padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;
                               box-shadow:0 2px 6px rgba(0,0,0,0.35);">{food['cuisine_label']}</span>
                  {super_badge}
                  <div style="padding:10px;color:white;">
                    <div style="font-weight:700;">{food['food_name']}</div>
                    <div style="color:#8A93B0;font-size:12px;">{food['description'][:50]}</div>
                    <div style="color:#FF6B2B;font-weight:700;margin-top:4px;">từ {food['price']:,} VNĐ</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Chọn món này", key=f"battle-pick-{food['id']}", use_container_width=True, type="primary"):
                st.session_state["winner"] = food
                goto("map")

    st.caption("Chạm vào món để tiếp tục tìm quán ăn")


# ─── Screen: MAP ────────────────────────────────────────────────
def render_map() -> None:
    winner = st.session_state["winner"]
    if not winner:
        goto("main")

    profile = st.session_state["profile"]
    ctx = st.session_state["context"]

    top_l, top_r = st.columns([1, 2])
    if top_l.button("← Quay lại", key="map-back"):
        goto("main")
    with top_r:
        st.markdown(
            f"""
            <div style="display:flex;justify-content:flex-end;">
              <div style="color:white;background:#1E2745;border:1px solid #FF6B2B;
                          border-radius:14px;padding:6px 12px;font-size:13px;">
                Bạn chọn: <b style="color:#FF6B2B;">{winner['food_name']}</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    restaurants = filter_and_sort_restaurants(
        food_df, winner["food_name"], ctx["budget"],
        profile["lat"], profile["lon"], ctx["time_available"],
    )

    if not restaurants:
        st.warning("Chưa có quán nào phục vụ món này trong khu vực. Thử tìm lại!")
        if st.button("🔄 Tìm lại", type="primary", use_container_width=True):
            goto("main")
        return

    fmap = build_folium_map(profile["lat"], profile["lon"], restaurants)
    st_folium(fmap, height=300, use_container_width=True, returned_objects=[])

    st.markdown("### Quán ăn gần bạn")
    st.caption(f"{len(restaurants)} quán phục vụ {winner['food_name']} • trong ngân sách")

    for i, r in enumerate(restaurants):
        highlighted = i == 0
        bg = "rgba(255,107,43,0.08)" if highlighted else "#1E2745"
        border = "#FF6B2B" if highlighted else "#2C3555"
        district = estimate_district(r["lat"], r["lon"])
        st.markdown(
            f"""
            <div style="border:1px solid {border};border-radius:12px;padding:14px;
                        background:{bg};margin:8px 0;color:white;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div style="font-weight:700;">{r['shop_name']}</div>
                <span style="background:{r['price_label_color']};color:white;border-radius:10px;
                             padding:2px 8px;font-size:11px;">{r['price_label']}</span>
              </div>
              <div style="color:#8A93B0;font-size:13px;margin-top:4px;">📍 {district}, HCM</div>
              <div style="color:#8A93B0;font-size:13px;">
                🗺️ {r['_distance_km']:.1f}km · ⏱️ ~{r['_walk_min']} phút · ★ {r['rating']}
              </div>
              <div style="margin-top:8px;color:#FF6B2B;font-weight:700;">{r['price']:,} VNĐ</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"Dẫn đường →  {r['shop_name']}", key=f"nav-{i}", use_container_width=True):
            st.session_state["selected_shop"] = r
            st.session_state["show_route_popup"] = True
            st.rerun()

    if st.session_state.get("show_route_popup") and st.session_state.get("selected_shop"):
        _render_route_popup()


def _render_route_popup() -> None:
    shop = st.session_state["selected_shop"]
    district = estimate_district(shop["lat"], shop["lon"])
    steps = generate_mock_directions(shop["_distance_km"], shop["shop_name"])

    steps_html = "".join(
        f"""<div style="display:flex;gap:10px;align-items:flex-start;padding:8px 0;
                    border-bottom:1px solid #2C3555;color:#e1e6f5;">
              <div style="background:#FF6B2B;color:white;width:24px;height:24px;border-radius:50%;
                          display:flex;align-items:center;justify-content:center;font-size:12px;
                          font-weight:700;flex-shrink:0;">{i}</div>
              <div>{step}</div>
            </div>"""
        for i, step in enumerate(steps, start=1)
    )

    st.markdown(
        f"""
        <div style="background:#1A2040;border-radius:16px;padding:16px;margin-top:14px;
                    border:1px solid #2C3555;color:white;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <b>🗺️ Dẫn đường tới</b><br>
              <span style="color:#FF6B2B;font-weight:700;">{shop['shop_name']}</span><br>
              <span style="color:#8A93B0;font-size:13px;">{district}, HCM</span>
            </div>
          </div>
          <div style="background:#12172B;border-radius:10px;padding:10px;margin:10px 0;
                      color:#cfd6f0;font-size:14px;text-align:center;">
            📍 {shop['_distance_km']:.1f}km · ⏱️ {shop['_walk_min']} phút · 🚶 Đi bộ
          </div>
          <div style="font-weight:700;margin-bottom:6px;">Hướng dẫn từng bước:</div>
          {steps_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 3])
    if c1.button("✕ Đóng", key="route-close"):
        st.session_state["show_route_popup"] = False
        st.rerun()
    if c2.button("✈️ Đã đến! Đánh giá ngay", type="primary",
                 use_container_width=True, key="route-arrived"):
        st.session_state["show_route_popup"] = False
        goto("rating")


# ─── Screen: RATING ─────────────────────────────────────────────
def render_rating() -> None:
    shop = st.session_state.get("selected_shop")
    winner = st.session_state.get("winner")
    if not shop or not winner:
        goto("main")

    st.markdown(
        "<div style='text-align:center;font-size:64px;margin-top:8px;'>😄</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="text-align:center;color:white;">
          <div style="color:#8A93B0;">Bữa ăn của bạn tại</div>
          <div style="color:#FF6B2B;font-weight:700;font-size:1.2rem;">{shop['shop_name']}</div>
          <div style="color:#8A93B0;">{winner['food_name']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Đánh giá để AI cải thiện gợi ý cho lần tới")

    star_cols = st.columns(5)
    rv = st.session_state["rating_value"]
    for i in range(1, 6):
        star = "⭐" if i <= rv else "☆"
        if star_cols[i - 1].button(star, key=f"star-{i}", use_container_width=True):
            st.session_state["rating_value"] = i
            st.rerun()

    st.markdown("**Thêm nhận xét:**")
    tag_options = ["Ngon lắm!", "Khá ngon", "Bình thường", "Không ngon", "Giá hợp lý", "Hơi đắt"]
    selected_tags = st.session_state["rating_tags"]
    rows = [tag_options[i : i + 3] for i in range(0, len(tag_options), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for col, tag in zip(cols, row):
            is_on = tag in selected_tags
            btn_type = "primary" if is_on else "secondary"
            if col.button(tag, key=f"tag-{tag}", type=btn_type, use_container_width=True):
                if is_on:
                    selected_tags.remove(tag)
                else:
                    selected_tags.append(tag)
                st.rerun()

    st.markdown(
        "<div style='background:#1E2745;border:1px solid #2C3555;border-radius:10px;padding:10px;"
        "color:#cfd6f0;margin:14px 0;font-size:13px;'>"
        "🤖 Đánh giá cập nhật Decision Tree model — AI học từ thói quen của bạn.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Gửi & Bắt đầu lại", type="primary", use_container_width=True, key="rating-submit"):
        if st.session_state["rating_value"] <= 0:
            st.warning("Hãy chọn số sao!")
            return
        _commit_rating()
        goto("rating_done")

    st.caption("Dữ liệu lưu vào session để huấn luyện mô hình")


def _commit_rating() -> None:
    profile = st.session_state["profile"]
    winner = st.session_state["winner"]
    shop = st.session_state["selected_shop"]
    rating = st.session_state["rating_value"]

    st.session_state["ml_features"].append(encode_food(winner))
    st.session_state["ml_labels"].append(1 if rating >= 3 else 0)

    profile["total_orders"] += 1
    if rating >= 4:
        profile["total_likes"] += 1
    new_avg = (profile["avg_rating"] * (profile["total_orders"] - 1) + rating) / profile["total_orders"]
    profile["avg_rating"] = round(new_avg, 1)

    st.session_state["history"].append({
        "food_name": winner["food_name"],
        "shop_name": shop["shop_name"],
        "price": int(shop["price"]),
        "rating": rating,
        "tags": list(st.session_state["rating_tags"]),
        "emoji": CUISINE_EMOJI.get(winner.get("cuisine"), "🍽️"),
        "timestamp": datetime.now().strftime("%d/%m %H:%M"),
    })


# ─── Screen: RATING DONE ────────────────────────────────────────
def render_rating_done() -> None:
    # Snapshot rating before we clear state (so it still renders after reset).
    rv = st.session_state["rating_value"]
    stars = "⭐" * rv + "☆" * (5 - rv)

    st.markdown(
        f"""
        <div style="text-align:center;margin-top:20px;color:white;">
          <div style="background:#FF6B2B;width:80px;height:80px;border-radius:50%;
                      display:flex;align-items:center;justify-content:center;font-size:46px;color:white;
                      margin:0 auto 16px;">✓</div>
          <h2 style="color:white;">Cảm ơn bạn! 🙏</h2>
          <p style="color:#8A93B0;">Đánh giá của bạn giúp AI học hỏi<br>
            và đưa ra gợi ý tốt hơn cho lần sau!</p>
          <div style="color:#FFD700;font-size:26px;margin:14px 0;">{stars}</div>
          <div style="background:#1E2745;border:1px solid #2C3555;border-radius:10px;padding:10px;
                      color:#cfd6f0;font-size:13px;display:inline-block;">
            🤖 Đã cập nhật Decision Tree model<br>
            để huấn luyện từ thói quen
          </div>
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )

    # Manual escape hatch — works regardless of auto-redirect.
    if st.button("Về trang chủ →", type="primary", use_container_width=True, key="rating-done-home"):
        _reset_after_rating()
        goto("main")

    # Auto-redirect after 3s. time.sleep blocks the script but Streamlit flushes
    # the markdown to the browser before the sleep completes, so users see the
    # thank-you screen during the wait.
    time.sleep(3)
    _reset_after_rating()
    goto("main")


def _reset_after_rating() -> None:
    st.session_state["rating_value"] = 0
    st.session_state["rating_tags"] = []
    st.session_state["winner"] = None
    st.session_state["selected_shop"] = None
    st.session_state["show_route_popup"] = False
    st.session_state["deck"] = []
    st.session_state["deck_index"] = 0
    st.session_state["liked"] = []
    st.session_state["disliked"] = []
    st.session_state["super_liked"] = []


# ─── Bootstrapping ──────────────────────────────────────────────
@st.cache_data
def cached_food_data() -> pd.DataFrame:
    return load_food_data()


@st.cache_resource
def cached_fis():
    return build_fuzzy_system()


def main() -> None:
    load_css()
    init_session_state()

    global food_df, fis_sim
    try:
        food_df = cached_food_data()
    except Exception as exc:
        st.error("Không tải được dữ liệu món ăn. Kiểm tra file food_data_mockup.csv")
        st.exception(exc)
        st.stop()
    fis_sim = cached_fis()

    screen = st.session_state.get("screen", "main")

    renderers = {
        "main":        render_main,
        "profile":     render_profile,
        "swipe":       render_swipe,
        "battle":      render_battle,
        "map":         render_map,
        "rating":      render_rating,
        "rating_done": render_rating_done,
    }
    renderer = renderers.get(screen)
    if renderer is None:
        st.session_state["screen"] = "main"
        st.rerun()
    else:
        renderer()


if __name__ == "__main__":
    main()
