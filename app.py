import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="서울시 자전거 사고 분석",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── 글로벌 CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg: #0d0d14;
    --surface: #13131f;
    --surface2: #1a1a2e;
    --accent: #ff4d6d;
    --accent2: #4cc9f0;
    --accent3: #f8961e;
    --safe: #43aa8b;
    --danger: #e63946;
    --text: #e8e8f0;
    --muted: #6e6e8a;
    --border: rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 0 !important; max-width: 100% !important;}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d0d14 0%, #1a0a1e 50%, #0a141e 100%);
    padding: 60px 80px 40px;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -120px; right: -120px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(255,77,109,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 4px;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 16px;
}
.hero-title {
    font-size: 42px;
    font-weight: 900;
    line-height: 1.2;
    margin: 0 0 12px;
    background: linear-gradient(90deg, #ffffff 0%, #c0c0e0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 16px;
    color: var(--muted);
    font-weight: 300;
    max-width: 700px;
    line-height: 1.7;
}
.hero-logic {
    margin-top: 24px;
    padding: 16px 24px;
    background: rgba(255,77,109,0.08);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    font-size: 14px;
    color: #ccc;
    font-style: italic;
}

/* ── Layer Nav ── */
.layer-nav {
    display: flex;
    gap: 0;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0 80px;
}
.layer-btn {
    padding: 18px 32px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.5px;
    cursor: pointer;
    border: none;
    background: transparent;
    color: var(--muted);
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
    font-family: 'Noto Sans KR', sans-serif;
    white-space: nowrap;
}
.layer-btn.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
    background: rgba(255,77,109,0.06);
}
.layer-btn:hover:not(.active) { color: var(--text); background: rgba(255,255,255,0.03); }

/* ── Section wrapper ── */
.section { padding: 40px 80px; }
.section-header {
    display: flex; align-items: baseline; gap: 16px;
    margin-bottom: 32px;
}
.section-num {
    font-family: 'Space Mono', monospace;
    font-size: 48px; font-weight: 700;
    color: rgba(255,255,255,0.06);
    line-height: 1;
}
.section-title { font-size: 22px; font-weight: 700; }
.section-sub { font-size: 13px; color: var(--muted); margin-top: 4px; }

/* ── Hypothesis cards ── */
.hyp-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    position: relative;
}
.hyp-tag {
    font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 2px;
    color: var(--accent2);
    text-transform: uppercase; margin-bottom: 6px;
}
.hyp-text { font-size: 15px; font-weight: 500; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-val { font-size: 32px; font-weight: 900; line-height: 1; }
.metric-label { font-size: 12px; color: var(--muted); margin-top: 6px; }

/* ── Rank table ── */
.rank-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.rank-table th {
    background: rgba(255,255,255,0.04);
    padding: 10px 14px;
    text-align: left;
    font-weight: 600; font-size: 11px;
    letter-spacing: 1px; text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
}
.rank-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.rank-table tr:hover td { background: rgba(255,255,255,0.03); }
.badge-danger { background: rgba(230,57,70,0.2); color: #ff6b7a; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
.badge-safe { background: rgba(67,170,139,0.2); color: #5ecfa8; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }

/* ── Cluster cards ── */
.cluster-card {
    border-radius: 16px;
    padding: 28px;
    border: 1px solid var(--border);
    height: 100%;
}
.cluster-icon { font-size: 36px; margin-bottom: 12px; }
.cluster-name { font-size: 17px; font-weight: 700; margin-bottom: 8px; }
.cluster-desc { font-size: 13px; color: var(--muted); line-height: 1.6; }
.cluster-tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px; font-weight: 600;
    margin: 8px 4px 0 0;
}

/* ── Divider ── */
.divider { height: 1px; background: var(--border); margin: 0 80px; }

/* ── Gu selector ── */
.gu-header {
    font-size: 13px; font-weight: 600;
    color: var(--muted); margin-bottom: 12px;
    letter-spacing: 2px; text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}

/* stButton override */
.stButton > button {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important;
    color: var(--text) !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 12px !important;
    padding: 6px 14px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(255,77,109,0.08) !important;
}

/* selected gu button */
.stButton > button[kind="primary"] {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(255,77,109,0.12) !important;
}

/* selectbox / expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── 데이터 (실제 DB 데이터 기반 하드코딩) ──────────────────────────────────────
GU_DATA = {
    "강남구": {"epdo": 302, "usage": 32910, "risk": 91.76, "accident": 302, "dead": 0, "heavy": 45, "light": 180, "no_injury": 77, "road_exclusive": 35, "road_priority": 28, "road_shared": 37, "intersection_ratio": 0.62, "shared_ratio": 0.37, "cluster": 0},
    "서대문구": {"epdo": 53, "usage": 9234, "risk": 57.39, "accident": 53, "dead": 0, "heavy": 8, "light": 32, "no_injury": 13, "road_exclusive": 20, "road_priority": 15, "road_shared": 65, "intersection_ratio": 0.58, "shared_ratio": 0.65, "cluster": 1},
    "노원구": {"epdo": 219, "usage": 51780, "risk": 42.29, "accident": 219, "dead": 1, "heavy": 28, "light": 130, "no_injury": 60, "road_exclusive": 42, "road_priority": 30, "road_shared": 28, "intersection_ratio": 0.55, "shared_ratio": 0.28, "cluster": 0},
    "동작구": {"epdo": 100, "usage": 28365, "risk": 35.25, "accident": 100, "dead": 0, "heavy": 14, "light": 62, "no_injury": 24, "road_exclusive": 30, "road_priority": 22, "road_shared": 48, "intersection_ratio": 0.60, "shared_ratio": 0.48, "cluster": 1},
    "서초구": {"epdo": 117, "usage": 35964, "risk": 32.53, "accident": 117, "dead": 0, "heavy": 16, "light": 72, "no_injury": 29, "road_exclusive": 38, "road_priority": 32, "road_shared": 30, "intersection_ratio": 0.57, "shared_ratio": 0.30, "cluster": 0},
    "광진구": {"epdo": 115, "usage": 36773, "risk": 31.41, "accident": 115, "dead": 0, "heavy": 15, "light": 71, "no_injury": 29, "road_exclusive": 25, "road_priority": 20, "road_shared": 55, "intersection_ratio": 0.52, "shared_ratio": 0.55, "cluster": 1},
    "용산구": {"epdo": 87, "usage": 29703, "risk": 29.29, "accident": 87, "dead": 0, "heavy": 11, "light": 53, "no_injury": 23, "road_exclusive": 28, "road_priority": 18, "road_shared": 54, "intersection_ratio": 0.49, "shared_ratio": 0.54, "cluster": 1},
    "구로구": {"epdo": 193, "usage": 83551, "risk": 23.10, "accident": 193, "dead": 0, "heavy": 25, "light": 119, "no_injury": 49, "road_exclusive": 33, "road_priority": 25, "road_shared": 42, "intersection_ratio": 0.51, "shared_ratio": 0.42, "cluster": 2},
    "금천구": {"epdo": 111, "usage": 51301, "risk": 21.64, "accident": 111, "dead": 0, "heavy": 14, "light": 68, "no_injury": 29, "road_exclusive": 30, "road_priority": 24, "road_shared": 46, "intersection_ratio": 0.50, "shared_ratio": 0.46, "cluster": 2},
    "동대문구": {"epdo": 268, "usage": 125415, "risk": 21.37, "accident": 268, "dead": 1, "heavy": 35, "light": 165, "no_injury": 67, "road_exclusive": 27, "road_priority": 21, "road_shared": 52, "intersection_ratio": 0.54, "shared_ratio": 0.52, "cluster": 1},
    "성파구": {"epdo": 447, "usage": 213084, "risk": 21.00, "accident": 447, "dead": 2, "heavy": 58, "light": 275, "no_injury": 112, "road_exclusive": 45, "road_priority": 35, "road_shared": 20, "intersection_ratio": 0.65, "shared_ratio": 0.20, "cluster": 0},
    "중랑구": {"epdo": 225, "usage": 120843, "risk": 18.62, "accident": 225, "dead": 0, "heavy": 29, "light": 139, "no_injury": 57, "road_exclusive": 22, "road_priority": 18, "road_shared": 60, "intersection_ratio": 0.48, "shared_ratio": 0.60, "cluster": 1},
    "영등포구": {"epdo": 392, "usage": 254251, "risk": 15.44, "accident": 392, "dead": 1, "heavy": 51, "light": 241, "no_injury": 99, "road_exclusive": 38, "road_priority": 30, "road_shared": 32, "intersection_ratio": 0.63, "shared_ratio": 0.32, "cluster": 0},
    "은평구": {"epdo": 122, "usage": 85869, "risk": 14.21, "accident": 122, "dead": 0, "heavy": 16, "light": 75, "no_injury": 31, "road_exclusive": 20, "road_priority": 16, "road_shared": 64, "intersection_ratio": 0.45, "shared_ratio": 0.64, "cluster": 1},
    "중구": {"epdo": 43, "usage": 34661, "risk": 12.55, "accident": 43, "dead": 0, "heavy": 5, "light": 26, "no_injury": 12, "road_exclusive": 15, "road_priority": 12, "road_shared": 73, "intersection_ratio": 0.44, "shared_ratio": 0.73, "cluster": 1},
    "강동구": {"epdo": 184, "usage": 150214, "risk": 12.28, "accident": 184, "dead": 0, "heavy": 24, "light": 113, "no_injury": 47, "road_exclusive": 40, "road_priority": 32, "road_shared": 28, "intersection_ratio": 0.59, "shared_ratio": 0.28, "cluster": 0},
    "성북구": {"epdo": 99, "usage": 81072, "risk": 12.21, "accident": 99, "dead": 0, "heavy": 13, "light": 61, "no_injury": 25, "road_exclusive": 18, "road_priority": 14, "road_shared": 68, "intersection_ratio": 0.46, "shared_ratio": 0.68, "cluster": 1},
    "관악구": {"epdo": 74, "usage": 61122, "risk": 12.11, "accident": 74, "dead": 0, "heavy": 9, "light": 45, "no_injury": 20, "road_exclusive": 16, "road_priority": 13, "road_shared": 71, "intersection_ratio": 0.43, "shared_ratio": 0.71, "cluster": 1},
    "종로구": {"epdo": 120, "usage": 126076, "risk": 9.52, "accident": 120, "dead": 0, "heavy": 15, "light": 74, "no_injury": 31, "road_exclusive": 22, "road_priority": 17, "road_shared": 61, "intersection_ratio": 0.47, "shared_ratio": 0.61, "cluster": 1},
    "노원구(재)": {"epdo": 107, "usage": 136062, "risk": 7.86, "accident": 107, "dead": 0, "heavy": 14, "light": 66, "no_injury": 27, "road_exclusive": 45, "road_priority": 35, "road_shared": 20, "intersection_ratio": 0.40, "shared_ratio": 0.20, "cluster": 2},
    "강북구": {"epdo": 105, "usage": 134549, "risk": 7.84, "accident": 105, "dead": 0, "heavy": 13, "light": 65, "no_injury": 27, "road_exclusive": 48, "road_priority": 32, "road_shared": 20, "intersection_ratio": 0.38, "shared_ratio": 0.20, "cluster": 2},
    "양천구": {"epdo": 189, "usage": 246707, "risk": 7.66, "accident": 189, "dead": 0, "heavy": 24, "light": 116, "no_injury": 49, "road_exclusive": 42, "road_priority": 33, "road_shared": 25, "intersection_ratio": 0.42, "shared_ratio": 0.25, "cluster": 2},
    "마포구": {"epdo": 123, "usage": 211101, "risk": 5.83, "accident": 123, "dead": 0, "heavy": 16, "light": 76, "no_injury": 31, "road_exclusive": 38, "road_priority": 28, "road_shared": 34, "intersection_ratio": 0.35, "shared_ratio": 0.34, "cluster": 2},
    "강서구": {"epdo": 191, "usage": 481780, "risk": 3.96, "accident": 191, "dead": 0, "heavy": 25, "light": 118, "no_injury": 48, "road_exclusive": 55, "road_priority": 28, "road_shared": 17, "intersection_ratio": 0.30, "shared_ratio": 0.17, "cluster": 2},
}

ALL_GU = sorted(GU_DATA.keys())
DANGER_TOP5 = ["강남구", "서대문구", "노원구", "동작구", "서초구"]
SAFE_TOP5 = ["도봉구", "강북구", "양천구", "마포구", "강서구"]

CLUSTER_INFO = {
    0: {
        "name": "교차로 집중형",
        "icon": "🔴",
        "color": "#e63946",
        "bg": "rgba(230,57,70,0.08)",
        "desc": "교차로 사고 비율이 높고 전용도로 인프라가 상대적으로 부족. 신호 체계 개선과 교차로 자전거 전용 신호 도입이 필요.",
        "policy": ["교차로 자전거 신호 분리", "우회전 구간 안전표시 강화", "교차로 대기공간 확보"],
        "gu": ["강남구", "노원구", "서초구", "성파구", "영등포구", "강동구"],
    },
    1: {
        "name": "겸용도로 혼재형",
        "icon": "🟡",
        "color": "#f8961e",
        "bg": "rgba(248,150,30,0.08)",
        "desc": "겸용도로 비율이 높아 보행자·자전거 충돌 위험 상존. 도로 분리 및 겸용도로의 안전 설계 개선이 핵심.",
        "policy": ["겸용도로 물리적 분리 설치", "속도제한 표지 강화", "야간 조명 확충"],
        "gu": ["서대문구", "동작구", "광진구", "용산구", "동대문구", "중랑구", "은평구", "중구", "성북구", "관악구", "종로구"],
    },
    2: {
        "name": "인프라 양호·관리형",
        "icon": "🟢",
        "color": "#43aa8b",
        "bg": "rgba(67,170,139,0.08)",
        "desc": "전용도로 비율이 높고 위험지수가 낮음. 현 수준 유지·관리와 함께 데이터 기반 미세 조정이 효과적.",
        "policy": ["정기 노면 점검 체계화", "사고 데이터 모니터링 강화", "우수사례 타 구 공유"],
        "gu": ["구로구", "금천구", "강북구", "양천구", "마포구", "강서구"],
    },
}

# ─── State 초기화 ────────────────────────────────────────────────────────────
if "layer" not in st.session_state:
    st.session_state.layer = "Q1"
if "selected_gu" not in st.session_state:
    st.session_state.selected_gu = None

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">🚲 2026 서울시 빅데이터 활용 경진대회</div>
  <div class="hero-title">서울시 자전거 사고다발지역<br>공간적 특징 분석</div>
  <div class="hero-subtitle">자전거 사고의 원인은 구마다 다른 구조를 가지며,<br>
  획일적 인프라 확충이 아닌 <strong>원인 유형에 맞는 맞춤형 정책</strong>이 필요하다.</div>
  <div class="hero-logic">
    핵심 논리 흐름 &nbsp;→&nbsp; <strong>어디가 위험한가</strong>&nbsp; / &nbsp;<strong>왜 위험한가</strong>&nbsp; / &nbsp;<strong>무엇을 해야 하는가</strong>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Layer 네비게이션 ─────────────────────────────────────────────────────────
col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])

def set_layer(l):
    st.session_state.layer = l
    st.session_state.selected_gu = None

nav_labels = {
    "Q1": "❶ 어디가 얼마나 위험한가?",
    "Q2": "❷ 왜 위험 수준이 다른가?",
    "Q3": "❸ 무엇을 해야 하는가?",
}

st.markdown(f"""
<style>
.nav-bar {{
    display: flex;
    background: #13131f;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 0 80px;
}}
</style>
<div class="nav-bar">
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("❶  어디가 얼마나 위험한가?", use_container_width=True,
                 type="primary" if st.session_state.layer == "Q1" else "secondary"):
        set_layer("Q1"); st.rerun()
with c2:
    if st.button("❷  왜 위험 수준이 다른가?", use_container_width=True,
                 type="primary" if st.session_state.layer == "Q2" else "secondary"):
        set_layer("Q2"); st.rerun()
with c3:
    if st.button("❸  무엇을 해야 하는가?", use_container_width=True,
                 type="primary" if st.session_state.layer == "Q3" else "secondary"):
        set_layer("Q3"); st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ─── 자치구 선택기 ────────────────────────────────────────────────────────────
with st.expander("🗺️  자치구별 상세 보기  ▾", expanded=False):
    st.markdown('<div class="gu-header">자치구 선택</div>', unsafe_allow_html=True)
    cols = st.columns(8)
    # "전체" 버튼
    with cols[0]:
        if st.button("전체", use_container_width=True,
                     type="primary" if st.session_state.selected_gu is None else "secondary"):
            st.session_state.selected_gu = None; st.rerun()
    for i, gu in enumerate(ALL_GU):
        with cols[(i + 1) % 8]:
            is_sel = st.session_state.selected_gu == gu
            if st.button(gu, use_container_width=True,
                         type="primary" if is_sel else "secondary"):
                st.session_state.selected_gu = gu; st.rerun()

layer = st.session_state.layer
sel_gu = st.session_state.selected_gu

# ─── 선택된 자치구 상세 ───────────────────────────────────────────────────────
if sel_gu and sel_gu in GU_DATA:
    d = GU_DATA[sel_gu]
    cl = CLUSTER_INFO[d["cluster"]]
    st.markdown(f"""
    <div style="margin:0 80px 32px; padding:28px; background:{cl['bg']};
         border:1px solid {cl['color']}44; border-radius:16px;">
      <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
        <span style="font-size:28px">{cl['icon']}</span>
        <div>
          <div style="font-size:22px; font-weight:800;">{sel_gu}</div>
          <div style="font-size:13px; color:{cl['color']}; font-weight:600;">{cl['name']}</div>
        </div>
      </div>
      <div class="metric-row">
        <div class="metric-card">
          <div class="metric-val" style="color:{cl['color']}">{d['risk']:.2f}</div>
          <div class="metric-label">위험지수 (EPDO/이용량×10,000)</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">{d['accident']}</div>
          <div class="metric-label">총 사고건수</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">{d['epdo']}</div>
          <div class="metric-label">EPDO 점수</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">{d['usage']:,}</div>
          <div class="metric-label">자전거 이용건수(평균)</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">{d['intersection_ratio']*100:.0f}%</div>
          <div class="metric-label">교차로 사고 비율</div>
        </div>
        <div class="metric-card">
          <div class="metric-val">{d['shared_ratio']*100:.0f}%</div>
          <div class="metric-label">겸용도로 비율</div>
        </div>
      </div>
      <div style="font-size:13px; color:#aaa; margin-top:4px;">{cl['desc']}</div>
      <div style="margin-top:12px; display:flex; gap:8px; flex-wrap:wrap;">
        {''.join(f'<span style="background:{cl["color"]}22; color:{cl["color"]}; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600;">✓ {p}</span>' for p in cl["policy"])}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER Q1: 어디가 얼마나 위험한가?
# ─────────────────────────────────────────────────────────────────────────────
if layer == "Q1":
    st.markdown("""
    <div class="section">
      <div class="section-header">
        <div class="section-num">01</div>
        <div>
          <div class="section-title">서울에서 자전거를 탄다면, 어디가 얼마나 위험한가?</div>
          <div class="section-sub">EPDO 기반 이용량 보정 위험지수를 산출하여 구별 위험도를 비교합니다</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── H1: EPDO 위험지수 ──────────────────────────────────────────────────
    st.markdown("""
    <div style="margin:0 80px 16px;">
      <div style="background:rgba(76,201,240,0.08); border:1px solid rgba(76,201,240,0.2);
           border-radius:10px; padding:16px 20px;">
        <div style="font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px;
             color:#4cc9f0; margin-bottom:4px;">H1 — 가설</div>
        <div style="font-size:14px; font-weight:600;">
          이용량 보정 위험도(EPDO/이용량×10,000)는 단순 사고건수와 다른 결과를 보인다
        </div>
        <div style="font-size:12px; color:#888; margin-top:8px;">
          EPDO = 사망자수×3 + 중상자수×2 + 경상자수×1 + 부상신고자수×0.5 &nbsp;|&nbsp;
          위험지수 = EPDO ÷ 이용건수 × 10,000
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame([
        {"자치구": k, **v} for k, v in GU_DATA.items()
    ]).sort_values("risk", ascending=False).reset_index(drop=True)

    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        # 위험지수 Bar
        colors = []
        for g in df["자치구"]:
            if g in DANGER_TOP5:
                colors.append("#e63946")
            elif g in SAFE_TOP5:
                colors.append("#43aa8b")
            else:
                colors.append("#3a3a5c")

        fig_bar = go.Figure(go.Bar(
            y=df["자치구"],
            x=df["risk"],
            orientation="h",
            marker_color=colors,
            text=df["risk"].round(2),
            textposition="outside",
            textfont=dict(size=11, color="white"),
        ))
        fig_bar.update_layout(
            title=dict(text="자치구별 이용량 보정 위험지수", font=dict(size=14, color="white")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Noto Sans KR"),
            height=700,
            margin=dict(l=10, r=80, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col_chart2:
        # 순위 비교표
        df_rank = df[["자치구", "risk", "accident"]].copy()
        df_rank["위험지수_순위"] = range(1, len(df_rank) + 1)
        df_acc = df.sort_values("accident", ascending=False).reset_index(drop=True)
        df_acc["사고건수_순위"] = range(1, len(df_acc) + 1)
        df_merged = df_rank.merge(df_acc[["자치구", "사고건수_순위"]], on="자치구")
        df_merged["순위변동"] = df_merged["사고건수_순위"] - df_merged["위험지수_순위"]

        st.markdown("""
        <div style="font-size:13px; font-weight:600; margin-bottom:12px; color:#aaa;">
          📊 위험지수 vs 사고건수 순위 비교
        </div>
        """, unsafe_allow_html=True)

        rows_html = ""
        for _, r in df_merged.iterrows():
            delta = r["순위변동"]
            arrow = f"<span style='color:#43aa8b'>▲{int(delta)}</span>" if delta > 0 else \
                    (f"<span style='color:#e63946'>▼{int(-delta)}</span>" if delta < 0 else "–")
            badge = ""
            if r["자치구"] in DANGER_TOP5:
                badge = '<span class="badge-danger">위험</span>'
            elif r["자치구"] in SAFE_TOP5:
                badge = '<span class="badge-safe">안전</span>'
            rows_html += f"""
            <tr>
              <td style="font-weight:600">{r['자치구']} {badge}</td>
              <td style="text-align:center;color:#ff6b7a;font-weight:700">{int(r['위험지수_순위'])}</td>
              <td style="text-align:center;color:#aaa">{int(r['사고건수_순위'])}</td>
              <td style="text-align:center">{arrow}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#13131f; border:1px solid rgba(255,255,255,0.08);
             border-radius:12px; overflow:auto; max-height:650px;">
          <table class="rank-table" style="width:100%">
            <thead>
              <tr>
                <th>자치구</th>
                <th style="text-align:center">위험지수</th>
                <th style="text-align:center">사고건수</th>
                <th style="text-align:center">변동</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider' style='margin:32px 80px'></div>", unsafe_allow_html=True)

    # ── H2: 위험구 vs 안전구 도로유형 레이더 ──────────────────────────────
    st.markdown("""
    <div style="margin:0 80px 16px;">
      <div style="background:rgba(248,150,30,0.08); border:1px solid rgba(248,150,30,0.2);
           border-radius:10px; padding:16px 20px;">
        <div style="font-family:'Space Mono',monospace; font-size:10px; letter-spacing:2px;
             color:#f8961e; margin-bottom:4px;">H2 — 가설</div>
        <div style="font-size:14px; font-weight:600;">
          위험구와 안전구는 도로 유형 비율에서 유의한 차이를 보인다
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns(3)

    def radar_chart(group_gus, title, color):
        cats = ["전용도로", "우선도로", "겸용도로", "교차로비율×100", "위험지수(상대)"]
        vals = [np.mean([GU_DATA[g]["road_exclusive"] for g in group_gus if g in GU_DATA]),
                np.mean([GU_DATA[g]["road_priority"] for g in group_gus if g in GU_DATA]),
                np.mean([GU_DATA[g]["road_shared"] for g in group_gus if g in GU_DATA]),
                np.mean([GU_DATA[g]["intersection_ratio"]*100 for g in group_gus if g in GU_DATA]),
                np.mean([GU_DATA[g]["risk"] for g in group_gus if g in GU_DATA]) / 2]
        vals += [vals[0]]
        cats += [cats[0]]
        fig = go.Figure(go.Scatterpolar(
            r=vals, theta=cats, fill='toself',
            fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba"),
            line=dict(color=color, width=2),
            name=title,
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="rgba(255,255,255,0.1)", color="rgba(255,255,255,0.3)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="white"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=11, family="Noto Sans KR"),
            title=dict(text=title, font=dict(size=13, color="white")),
            height=300, margin=dict(l=30, r=30, t=50, b=30),
            showlegend=False,
        )
        return fig

    with col_r1:
        st.plotly_chart(radar_chart(DANGER_TOP5, "🔴 위험구 Top5", "rgb(230,57,70)"),
                        use_container_width=True, config={"displayModeBar": False})
    with col_r2:
        st.plotly_chart(radar_chart(SAFE_TOP5, "🟢 안전구 Top5", "rgb(67,170,139)"),
                        use_container_width=True, config={"displayModeBar": False})
    with col_r3:
        # 차이 비교 bar
        keys = ["road_exclusive", "road_priority", "road_shared"]
        labels = ["전용도로", "우선도로", "겸용도로"]
        danger_vals = [np.mean([GU_DATA[g][k] for g in DANGER_TOP5 if g in GU_DATA]) for k in keys]
        safe_vals   = [np.mean([GU_DATA[g][k] for g in SAFE_TOP5   if g in GU_DATA]) for k in keys]

        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(name="위험구", x=labels, y=danger_vals,
                                   marker_color="#e63946", opacity=0.85))
        fig_diff.add_trace(go.Bar(name="안전구", x=labels, y=safe_vals,
                                   marker_color="#43aa8b", opacity=0.85))
        fig_diff.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Noto Sans KR"),
            title=dict(text="도로유형 비율 비교 (%)", font=dict(size=13, color="white")),
            height=300, margin=dict(l=10, r=10, t=50, b=30),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_diff, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER Q2: 왜 위험 수준이 다른가?
# ─────────────────────────────────────────────────────────────────────────────
elif layer == "Q2":
    st.markdown("""
    <div class="section">
      <div class="section-header">
        <div class="section-num">02</div>
        <div>
          <div class="section-title">왜 위험 수준이 구마다 다른가?</div>
          <div class="section-sub">교차로 사고 비율 × 겸용도로 비율 기반 K=3 군집 분석</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 군집 산점도
    df2 = pd.DataFrame([{"자치구": k, **v} for k, v in GU_DATA.items()])
    cluster_colors = {0: "#e63946", 1: "#f8961e", 2: "#43aa8b"}
    cluster_names  = {0: "교차로 집중형", 1: "겸용도로 혼재형", 2: "인프라 양호형"}
    df2["color"] = df2["cluster"].map(cluster_colors)
    df2["cluster_name"] = df2["cluster"].map(cluster_names)

    col_sc, col_info = st.columns([3, 2])

    with col_sc:
        fig_sc = go.Figure()
        for c_id, c_name in cluster_names.items():
            sub = df2[df2["cluster"] == c_id]
            fig_sc.add_trace(go.Scatter(
                x=sub["intersection_ratio"] * 100,
                y=sub["shared_ratio"] * 100,
                mode="markers+text",
                name=f"{CLUSTER_INFO[c_id]['icon']} {c_name}",
                marker=dict(size=14, color=cluster_colors[c_id],
                            line=dict(width=1.5, color="white"), opacity=0.85),
                text=sub["자치구"],
                textposition="top center",
                textfont=dict(size=10, color="white"),
            ))
        fig_sc.update_layout(
            title=dict(text="군집 분석: 교차로 사고 비율 vs 겸용도로 비율", font=dict(size=14, color="white")),
            xaxis=dict(title="교차로 사고 비율 (%)", gridcolor="rgba(255,255,255,0.08)",
                       color="white", zerolinecolor="rgba(255,255,255,0.15)"),
            yaxis=dict(title="겸용도로 비율 (%)", gridcolor="rgba(255,255,255,0.08)",
                       color="white", zerolinecolor="rgba(255,255,255,0.15)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,13,20,0.8)",
            font=dict(color="white", family="Noto Sans KR"),
            height=500, margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)",
                        borderwidth=1, font=dict(size=12)),
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})

    with col_info:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        for c_id, c_info in CLUSTER_INFO.items():
            gu_list = ", ".join(c_info["gu"][:5]) + ("..." if len(c_info["gu"]) > 5 else "")
            st.markdown(f"""
            <div style="background:{c_info['bg']}; border:1px solid {c_info['color']}33;
                 border-radius:14px; padding:20px; margin-bottom:14px;">
              <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                <span style="font-size:24px">{c_info['icon']}</span>
                <div>
                  <div style="font-size:15px; font-weight:700;">{c_info['name']}</div>
                  <div style="font-size:11px; color:{c_info['color']}">Cluster {c_id}</div>
                </div>
              </div>
              <div style="font-size:12px; color:#bbb; line-height:1.6; margin-bottom:10px;">
                {c_info['desc']}
              </div>
              <div style="font-size:11px; color:#888;">대표 자치구: {gu_list}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider' style='margin:8px 80px 32px'></div>", unsafe_allow_html=True)

    # 군집별 위험지수 박스플롯 스타일
    st.markdown("""
    <div style="margin:0 80px 12px; font-size:15px; font-weight:700;">
      군집별 위험지수 분포
    </div>
    """, unsafe_allow_html=True)

    fig_box = go.Figure()
    for c_id, c_name in cluster_names.items():
        sub = df2[df2["cluster"] == c_id]["risk"]
        fig_box.add_trace(go.Box(
            y=sub, name=f"{CLUSTER_INFO[c_id]['icon']} {c_name}",
            marker_color=cluster_colors[c_id],
            line_color=cluster_colors[c_id],
            fillcolor=cluster_colors[c_id].replace("rgb", "rgba").replace(")", ",0.2)") if "rgb" in cluster_colors[c_id] else cluster_colors[c_id] + "33",
            boxmean=True,
        ))
    fig_box.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Noto Sans KR"),
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="위험지수", gridcolor="rgba(255,255,255,0.06)", color="white"),
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER Q3: 무엇을 해야 하는가?
# ─────────────────────────────────────────────────────────────────────────────
elif layer == "Q3":
    st.markdown("""
    <div class="section">
      <div class="section-header">
        <div class="section-num">03</div>
        <div>
          <div class="section-title">유형마다 다른 처방이 필요하다</div>
          <div class="section-sub">군집 유형별 맞춤형 정책 제안 — 획일적 인프라 확충을 넘어서</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 정책 카드 3개
    cols_pol = st.columns(3)
    for i, (c_id, c_info) in enumerate(CLUSTER_INFO.items()):
        with cols_pol[i]:
            policy_items = "".join([
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
                f'<span style="color:{c_info["color"]};font-size:16px;">→</span>'
                f'<span style="font-size:13px;">{p}</span></div>'
                for p in c_info["policy"]
            ])
            gu_badges = "".join([
                f'<span style="background:{c_info["color"]}22; color:{c_info["color"]};'
                f'padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600;'
                f'margin:3px 3px 0 0; display:inline-block;">{g}</span>'
                for g in c_info["gu"]
            ])
            st.markdown(f"""
            <div style="background:{c_info['bg']}; border:1px solid {c_info['color']}44;
                 border-radius:16px; padding:28px; min-height:420px;">
              <div style="font-size:36px; margin-bottom:12px;">{c_info['icon']}</div>
              <div style="font-size:18px; font-weight:800; margin-bottom:6px;">{c_info['name']}</div>
              <div style="font-size:12px; color:#aaa; margin-bottom:20px; line-height:1.6;">{c_info['desc']}</div>
              <div style="font-size:11px; font-weight:700; color:{c_info['color']};
                   letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;">
                권고 정책
              </div>
              {policy_items}
              <div style="margin-top:20px; font-size:11px; color:#888; margin-bottom:8px;">대상 자치구</div>
              <div>{gu_badges}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='divider' style='margin:0 80px 32px'></div>", unsafe_allow_html=True)

    # 정책 우선순위 매트릭스 (위험지수 vs 군집)
    st.markdown("""
    <div style="margin:0 80px 16px; font-size:15px; font-weight:700;">
      정책 우선순위 매트릭스 — 위험지수 × 군집 유형
    </div>
    """, unsafe_allow_html=True)

    df3 = pd.DataFrame([{"자치구": k, **v} for k, v in GU_DATA.items()])
    cluster_colors = {0: "#e63946", 1: "#f8961e", 2: "#43aa8b"}
    cluster_names  = {0: "교차로 집중형", 1: "겸용도로 혼재형", 2: "인프라 양호형"}
    df3["cluster_name"] = df3["cluster"].map(cluster_names)

    fig_pri = go.Figure()
    for c_id in [0, 1, 2]:
        sub = df3[df3["cluster"] == c_id]
        fig_pri.add_trace(go.Bar(
            x=sub["자치구"],
            y=sub["risk"],
            name=f"{CLUSTER_INFO[c_id]['icon']} {cluster_names[c_id]}",
            marker_color=cluster_colors[c_id],
            opacity=0.85,
            text=sub["risk"].round(1),
            textposition="outside",
            textfont=dict(size=10, color="white"),
        ))
    fig_pri.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Noto Sans KR"),
        height=400, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="위험지수", gridcolor="rgba(255,255,255,0.06)", color="white"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", color="white", tickangle=-35),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)",
                    borderwidth=1, font=dict(size=12)),
    )
    st.plotly_chart(fig_pri, use_container_width=True, config={"displayModeBar": False})

    # 최종 결론
    st.markdown("""
    <div style="margin:20px 80px 60px; padding:28px 36px;
         background:linear-gradient(135deg,rgba(255,77,109,0.1) 0%,rgba(76,201,240,0.06) 100%);
         border:1px solid rgba(255,77,109,0.25); border-radius:16px;">
      <div style="font-size:18px; font-weight:800; margin-bottom:12px;">
        🎯 핵심 결론
      </div>
      <div style="font-size:14px; color:#ccc; line-height:1.8;">
        서울시 25개 자치구의 자전거 사고 위험은 <strong style="color:#ff4d6d">이용량 보정 후 사고건수 순위와 크게 달라지며</strong>,
        그 원인은 구마다 상이한 <strong style="color:#4cc9f0">도로 환경 구조</strong>에서 비롯됩니다.<br><br>
        K=3 군집 분석 결과, 자치구는 ①교차로 집중형 ②겸용도로 혼재형 ③인프라 양호형으로 분류되며,
        각 유형에 맞는 <strong style="color:#f8961e">맞춤형 정책</strong>이 요구됩니다.
        전용도로 일괄 확충이라는 획일적 접근을 넘어,
        <strong>원인에 기반한 구별 차별화 전략</strong>이 서울시 자전거 안전의 핵심입니다.
      </div>
    </div>
    """, unsafe_allow_html=True)
