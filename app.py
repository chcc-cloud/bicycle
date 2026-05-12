import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import os

# ─── 헬퍼: hex → rgba 문자열 (fillcolor 버그 완전 차단) ─────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─── 헬퍼: 신뢰 타원 좌표 계산 ──────────────────────────────────────────
def get_ellipse_points(x, y, n_std=1.5, n_points=100):
    if len(x) < 3:
        return [], []
    cov = np.cov(x, y)
    val, vec = np.linalg.eigh(cov)
    order = val.argsort()[::-1]
    val, vec = val[order], vec[:, order]
    theta = np.degrees(np.arctan2(*vec[:, 0][::-1]))
    t = np.linspace(0, 2 * np.pi, n_points)
    a = np.sqrt(np.maximum(val[0], 0)) * n_std
    b = np.sqrt(np.maximum(val[1], 0)) * n_std
    ell_x = a * np.cos(t)
    ell_y = b * np.sin(t)
    rot = np.array([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
        [np.sin(np.radians(theta)),  np.cos(np.radians(theta))],
    ])
    ell_r = rot @ np.vstack((ell_x, ell_y))
    return np.mean(x) + ell_r[0], np.mean(y) + ell_r[1]


# ─── DB 연결 ────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "bicycle.db")

@st.cache_data(ttl=300)
def load_db():
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            df_usage = pd.read_sql(
                'SELECT 자치구, "이용건수(평균)" as usage_avg '
                'FROM "서울시_23,24_자치구별_자전거_이용건수"', conn)
        except Exception:
            df_usage = None

        try:
            df_acc = pd.read_sql("""
                SELECT 자치구,
                       SUM(사고건수)    AS total_accident,
                       SUM(사망자수)    AS total_dead,
                       SUM(중상자수)    AS total_heavy,
                       SUM(경상자수)    AS total_light,
                       SUM(부상신고자수) AS total_no_injury,
                       AVG(CASE WHEN is_intersection='교차로' THEN 1.0 ELSE 0.0 END) AS intersection_ratio
                FROM "23_24_교차로사고다발지역"
                GROUP BY 자치구""", conn)
        except Exception:
            df_acc = None

        try:
            df_road = pd.read_sql("""
                SELECT 자치구, 계 AS total_road,
                       전용도로, 전용차로, 우선도로,
                       "경용도로(분리형)"   AS shared_sep,
                       "경용도로(비분리형)" AS shared_nonsep
                FROM "서울시_자전거 도로 유형별 자치구별 통계" """, conn)
        except Exception:
            df_road = None

        conn.close()
        if df_acc is None or df_usage is None:
            return None

        df = df_acc.merge(df_usage, on="자치구", how="left")
        if df_road is not None:
            df = df.merge(df_road, on="자치구", how="left")
            tot = df["total_road"].replace(0, 1).fillna(1)
            df["road_exclusive"] = ((df["전용도로"].fillna(0) + df["전용차로"].fillna(0)) / tot * 100).round(1)
            df["road_priority"]  = (df["우선도로"].fillna(0) / tot * 100).round(1)
            df["road_shared"]    = ((df["shared_sep"].fillna(0) + df["shared_nonsep"].fillna(0)) / tot * 100).round(1)
        else:
            df["road_exclusive"] = 30.0
            df["road_priority"]  = 20.0
            df["road_shared"]    = 50.0

        df["epdo"]  = (df["total_dead"]*3 + df["total_heavy"]*2
                       + df["total_light"]*1 + df["total_no_injury"]*0.5).round(0).astype(int)
        df["usage"] = df["usage_avg"].fillna(1).astype(int)
        df["risk"]  = (df["epdo"] / df["usage"].replace(0, 1) * 10000).round(2)
        df["intersection_ratio"] = df["intersection_ratio"].fillna(0.5)
        df["shared_ratio"] = (df["road_shared"] / 100).round(3)

        def _cluster(row):
            if row["intersection_ratio"] >= 0.55 and row["road_shared"] < 40:
                return 0
            elif row["road_shared"] >= 50:
                return 1
            else:
                return 2
        df["cluster"] = df.apply(_cluster, axis=1)

        result = {}
        for _, r in df.iterrows():
            result[r["자치구"]] = {
                "epdo": int(r["epdo"]), "usage": int(r["usage"]),
                "risk": float(r["risk"]), "accident": int(r["total_accident"]),
                "dead": int(r["total_dead"]), "heavy": int(r["total_heavy"]),
                "light": int(r["total_light"]), "no_injury": int(r["total_no_injury"]),
                "road_exclusive": float(r["road_exclusive"]),
                "road_priority":  float(r["road_priority"]),
                "road_shared":    float(r["road_shared"]),
                "intersection_ratio": float(r["intersection_ratio"]),
                "shared_ratio": float(r["shared_ratio"]),
                "cluster": int(r["cluster"]),
            }
        return result
    except Exception as e:
        st.sidebar.error(f"DB 오류: {e}")
        return None


# ─── Fallback 데이터 ──────────────────────────────────────────────────────
FALLBACK_DATA = {
    "강남구":   {"epdo":302,"usage":32910,"risk":91.76,"accident":302,"dead":0,"heavy":45,"light":180,"no_injury":77,"road_exclusive":35,"road_priority":28,"road_shared":37,"intersection_ratio":0.62,"shared_ratio":0.37,"cluster":0},
    "서대문구": {"epdo":53,"usage":9234,"risk":57.39,"accident":53,"dead":0,"heavy":8,"light":32,"no_injury":13,"road_exclusive":20,"road_priority":15,"road_shared":65,"intersection_ratio":0.58,"shared_ratio":0.65,"cluster":1},
    "노원구":   {"epdo":219,"usage":51780,"risk":42.29,"accident":219,"dead":1,"heavy":28,"light":130,"no_injury":60,"road_exclusive":42,"road_priority":30,"road_shared":28,"intersection_ratio":0.55,"shared_ratio":0.28,"cluster":0},
    "동작구":   {"epdo":100,"usage":28365,"risk":35.25,"accident":100,"dead":0,"heavy":14,"light":62,"no_injury":24,"road_exclusive":30,"road_priority":22,"road_shared":48,"intersection_ratio":0.60,"shared_ratio":0.48,"cluster":1},
    "서초구":   {"epdo":117,"usage":35964,"risk":32.53,"accident":117,"dead":0,"heavy":16,"light":72,"no_injury":29,"road_exclusive":38,"road_priority":32,"road_shared":30,"intersection_ratio":0.57,"shared_ratio":0.30,"cluster":0},
    "광진구":   {"epdo":115,"usage":36773,"risk":31.41,"accident":115,"dead":0,"heavy":15,"light":71,"no_injury":29,"road_exclusive":25,"road_priority":20,"road_shared":55,"intersection_ratio":0.52,"shared_ratio":0.55,"cluster":1},
    "용산구":   {"epdo":87,"usage":29703,"risk":29.29,"accident":87,"dead":0,"heavy":11,"light":53,"no_injury":23,"road_exclusive":28,"road_priority":18,"road_shared":54,"intersection_ratio":0.49,"shared_ratio":0.54,"cluster":1},
    "구로구":   {"epdo":193,"usage":83551,"risk":23.10,"accident":193,"dead":0,"heavy":25,"light":119,"no_injury":49,"road_exclusive":33,"road_priority":25,"road_shared":42,"intersection_ratio":0.51,"shared_ratio":0.42,"cluster":2},
    "금천구":   {"epdo":111,"usage":51301,"risk":21.64,"accident":111,"dead":0,"heavy":14,"light":68,"no_injury":29,"road_exclusive":30,"road_priority":24,"road_shared":46,"intersection_ratio":0.50,"shared_ratio":0.46,"cluster":2},
    "동대문구": {"epdo":268,"usage":125415,"risk":21.37,"accident":268,"dead":1,"heavy":35,"light":165,"no_injury":67,"road_exclusive":27,"road_priority":21,"road_shared":52,"intersection_ratio":0.54,"shared_ratio":0.52,"cluster":1},
    "송파구":   {"epdo":447,"usage":213084,"risk":21.00,"accident":447,"dead":2,"heavy":58,"light":275,"no_injury":112,"road_exclusive":45,"road_priority":35,"road_shared":20,"intersection_ratio":0.65,"shared_ratio":0.20,"cluster":0},
    "중랑구":   {"epdo":225,"usage":120843,"risk":18.62,"accident":225,"dead":0,"heavy":29,"light":139,"no_injury":57,"road_exclusive":22,"road_priority":18,"road_shared":60,"intersection_ratio":0.48,"shared_ratio":0.60,"cluster":1},
    "영등포구": {"epdo":392,"usage":254251,"risk":15.44,"accident":392,"dead":1,"heavy":51,"light":241,"no_injury":99,"road_exclusive":38,"road_priority":30,"road_shared":32,"intersection_ratio":0.63,"shared_ratio":0.32,"cluster":0},
    "은평구":   {"epdo":122,"usage":85869,"risk":14.21,"accident":122,"dead":0,"heavy":16,"light":75,"no_injury":31,"road_exclusive":20,"road_priority":16,"road_shared":64,"intersection_ratio":0.45,"shared_ratio":0.64,"cluster":1},
    "중구":     {"epdo":43,"usage":34661,"risk":12.55,"accident":43,"dead":0,"heavy":5,"light":26,"no_injury":12,"road_exclusive":15,"road_priority":12,"road_shared":73,"intersection_ratio":0.44,"shared_ratio":0.73,"cluster":1},
    "강동구":   {"epdo":184,"usage":150214,"risk":12.28,"accident":184,"dead":0,"heavy":24,"light":113,"no_injury":47,"road_exclusive":40,"road_priority":32,"road_shared":28,"intersection_ratio":0.59,"shared_ratio":0.28,"cluster":0},
    "성북구":   {"epdo":99,"usage":81072,"risk":12.21,"accident":99,"dead":0,"heavy":13,"light":61,"no_injury":25,"road_exclusive":18,"road_priority":14,"road_shared":68,"intersection_ratio":0.46,"shared_ratio":0.68,"cluster":1},
    "관악구":   {"epdo":74,"usage":61122,"risk":12.11,"accident":74,"dead":0,"heavy":9,"light":45,"no_injury":20,"road_exclusive":16,"road_priority":13,"road_shared":71,"intersection_ratio":0.43,"shared_ratio":0.71,"cluster":1},
    "종로구":   {"epdo":120,"usage":126076,"risk":9.52,"accident":120,"dead":0,"heavy":15,"light":74,"no_injury":31,"road_exclusive":22,"road_priority":17,"road_shared":61,"intersection_ratio":0.47,"shared_ratio":0.61,"cluster":1},
    "도봉구":   {"epdo":107,"usage":136062,"risk":7.86,"accident":107,"dead":0,"heavy":14,"light":66,"no_injury":27,"road_exclusive":45,"road_priority":35,"road_shared":20,"intersection_ratio":0.40,"shared_ratio":0.20,"cluster":2},
    "강북구":   {"epdo":105,"usage":134549,"risk":7.84,"accident":105,"dead":0,"heavy":13,"light":65,"no_injury":27,"road_exclusive":48,"road_priority":32,"road_shared":20,"intersection_ratio":0.38,"shared_ratio":0.20,"cluster":2},
    "양천구":   {"epdo":189,"usage":246707,"risk":7.66,"accident":189,"dead":0,"heavy":24,"light":116,"no_injury":49,"road_exclusive":42,"road_priority":33,"road_shared":25,"intersection_ratio":0.42,"shared_ratio":0.25,"cluster":2},
    "마포구":   {"epdo":123,"usage":211101,"risk":5.83,"accident":123,"dead":0,"heavy":16,"light":76,"no_injury":31,"road_exclusive":38,"road_priority":28,"road_shared":34,"intersection_ratio":0.35,"shared_ratio":0.34,"cluster":2},
    "강서구":   {"epdo":191,"usage":481780,"risk":3.96,"accident":191,"dead":0,"heavy":25,"light":118,"no_injury":48,"road_exclusive":55,"road_priority":28,"road_shared":17,"intersection_ratio":0.30,"shared_ratio":0.17,"cluster":2},
}

# ─── 데이터 결정 ──────────────────────────────────────────────────────────
db_data  = load_db()
GU_DATA  = db_data if db_data else FALLBACK_DATA

_sorted_risk = sorted(GU_DATA.items(), key=lambda x: x[1]["risk"], reverse=True)
DANGER_TOP5  = [g for g, _ in _sorted_risk[:5]]
SAFE_TOP5    = [g for g, _ in _sorted_risk[-5:]]
ALL_GU       = sorted(GU_DATA.keys())

# 군집 색상 (이미지와 동일: 빨강/주황/초록)
C_COLOR = {0: "#e63946", 1: "#f8961e", 2: "#43aa8b"}
C_FILL  = {0: "rgba(230,57,70,0.15)", 1: "rgba(248,150,30,0.15)", 2: "rgba(67,170,139,0.15)"}
C_LABEL = {0: "군집 3: 교차로 집중 위험형", 1: "군집 2: 혼합 위험형", 2: "군집 1: 저위험 인프라형"}

CLUSTER_INFO = {
    0: {
        "name": "교차로 집중 위험형",
        "icon": "🔴", "color": C_COLOR[0], "bg": "rgba(230,57,70,0.08)",
        "desc": "교차로 사고 비율이 높고 전용도로 인프라가 상대적으로 부족. 신호 체계 개선과 교차로 자전거 전용 신호 도입이 필요.",
        "policy": ["교차로 자전거 신호 분리", "우회전 구간 안전표시 강화", "교차로 대기공간 확보"],
        "gu": [g for g, v in GU_DATA.items() if v["cluster"] == 0],
    },
    1: {
        "name": "혼합 위험형",
        "icon": "🟡", "color": C_COLOR[1], "bg": "rgba(248,150,30,0.08)",
        "desc": "겸용도로 비율이 높아 보행자·자전거 충돌 위험 상존. 도로 분리 및 겸용도로의 안전 설계 개선이 핵심.",
        "policy": ["겸용도로 물리적 분리 설치", "속도제한 표지 강화", "야간 조명 확충"],
        "gu": [g for g, v in GU_DATA.items() if v["cluster"] == 1],
    },
    2: {
        "name": "저위험 인프라형",
        "icon": "🟢", "color": C_COLOR[2], "bg": "rgba(67,170,139,0.08)",
        "desc": "전용도로 비율이 높고 위험지수가 낮음. 현 수준 유지·관리와 함께 데이터 기반 미세 조정이 효과적.",
        "policy": ["정기 노면 점검 체계화", "사고 데이터 모니터링 강화", "우수사례 타 구 공유"],
        "gu": [g for g, v in GU_DATA.items() if v["cluster"] == 2],
    },
}

# ─── 페이지 설정 ────────────────────────────────────────────────────────────
st.set_page_config(page_title="서울시 자전거 사고 분석", page_icon="🚲",
                   layout="wide", initial_sidebar_state="collapsed")

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Space+Mono:wght@400;700&display=swap');
:root{--bg:#0d0d14;--surface:#13131f;--surface2:#1a1a2e;--accent:#ff4d6d;--accent2:#4cc9f0;--safe:#43aa8b;--danger:#e63946;--text:#e8e8f0;--muted:#6e6e8a;--border:rgba(255,255,255,0.08);}
html,body,[class*="css"]{font-family:'Noto Sans KR',sans-serif;background-color:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}

.hero{background:linear-gradient(135deg,#0d0d14 0%,#1a0a1e 50%,#0a141e 100%);padding:60px 80px 40px;border-bottom:1px solid var(--border);position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-120px;right:-120px;width:500px;height:500px;background:radial-gradient(circle,rgba(255,77,109,0.12) 0%,transparent 70%);pointer-events:none;}
.hero-label{font-family:'Space Mono',monospace;font-size:11px;letter-spacing:4px;color:var(--accent);text-transform:uppercase;margin-bottom:16px;}
.hero-title{font-size:42px;font-weight:900;line-height:1.2;margin:0 0 12px;background:linear-gradient(90deg,#ffffff 0%,#c0c0e0 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero-subtitle{font-size:16px;color:var(--muted);font-weight:300;max-width:700px;line-height:1.7;}
.hero-logic{margin-top:24px;padding:16px 24px;background:rgba(255,77,109,0.08);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;font-size:14px;color:#ccc;font-style:italic;}

.section{padding:40px 80px;}
.section-header{display:flex;align-items:baseline;gap:16px;margin-bottom:32px;}
.section-num{font-family:'Space Mono',monospace;font-size:48px;font-weight:700;color:rgba(255,255,255,0.06);line-height:1;}
.section-title{font-size:22px;font-weight:700;}
.section-sub{font-size:13px;color:var(--muted);margin-top:4px;}

.metric-row{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;}
.metric-card{flex:1;min-width:140px;background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center;}
.metric-val{font-size:32px;font-weight:900;line-height:1;}
.metric-label{font-size:12px;color:var(--muted);margin-top:6px;}

.rank-table{width:100%;border-collapse:collapse;font-size:13px;}
.rank-table th{background:rgba(255,255,255,0.04);padding:10px 14px;text-align:left;font-weight:600;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);}
.rank-table td{padding:10px 14px;border-bottom:1px solid rgba(255,255,255,0.04);}
.rank-table tr:hover td{background:rgba(255,255,255,0.03);}
.badge-danger{background:rgba(230,57,70,0.2);color:#ff6b7a;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;}
.badge-safe{background:rgba(67,170,139,0.2);color:#5ecfa8;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;}

.stButton>button{border-radius:8px!important;border:1px solid var(--border)!important;background:var(--surface2)!important;color:var(--text)!important;font-family:'Noto Sans KR',sans-serif!important;font-size:12px!important;padding:6px 14px!important;transition:all 0.15s!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;background:rgba(255,77,109,0.08)!important;}
.stButton>button[kind="primary"]{border-color:var(--accent)!important;color:var(--accent)!important;background:rgba(255,77,109,0.12)!important;}
.streamlit-expanderHeader{background:var(--surface2)!important;color:var(--text)!important;border-radius:10px!important;border:1px solid var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ─── State ──────────────────────────────────────────────────────────────────
if "layer" not in st.session_state:
    st.session_state.layer = "Q1"
if "selected_gu" not in st.session_state:
    st.session_state.selected_gu = None

# ─── DB 상태 ────────────────────────────────────────────────────────────────
if db_data:
    st.sidebar.success(f"✅ bicycle.db 연결됨 ({len(GU_DATA)}개 자치구)")
else:
    st.sidebar.warning("⚠️ bicycle.db 미연결 — 샘플 데이터 사용 중")

# ─── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">🚲 2026 서울시 빅데이터 활용 경진대회</div>
  <div class="hero-title">서울시 자전거 사고다발지역<br>공간적 특징 분석</div>
  <div class="hero-subtitle">자전거 사고의 원인은 구마다 다른 구조를 가지며,<br>
  획일적 인프라 확충이 아닌 <strong>원인 유형에 맞는 맞춤형 정책</strong>이 필요하다.</div>
  <div class="hero-logic">핵심 논리 흐름 &nbsp;→&nbsp; <strong>어디가 위험한가</strong>&nbsp;/&nbsp;<strong>왜 위험한가</strong>&nbsp;/&nbsp;<strong>무엇을 해야 하는가</strong></div>
</div>
""", unsafe_allow_html=True)

# ─── Layer 네비 ─────────────────────────────────────────────────────────────
def set_layer(l):
    st.session_state.layer = l
    st.session_state.selected_gu = None

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

# ─── 자치구 선택기 ───────────────────────────────────────────────────────────
with st.expander("🗺️  자치구별 상세 보기  ▾", expanded=False):
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:11px;letter-spacing:2px;color:#6e6e8a;margin-bottom:12px;">SELECT GU</div>', unsafe_allow_html=True)
    btn_cols = st.columns(8)
    with btn_cols[0]:
        if st.button("전체", use_container_width=True,
                     type="primary" if st.session_state.selected_gu is None else "secondary"):
            st.session_state.selected_gu = None; st.rerun()
    for i, gu in enumerate(ALL_GU):
        with btn_cols[(i + 1) % 8]:
            if st.button(gu, use_container_width=True,
                         type="primary" if st.session_state.selected_gu == gu else "secondary"):
                st.session_state.selected_gu = gu; st.rerun()

layer  = st.session_state.layer
sel_gu = st.session_state.selected_gu

# ─── 선택 자치구 패널 ────────────────────────────────────────────────────────
if sel_gu and sel_gu in GU_DATA:
    d  = GU_DATA[sel_gu]
    cl = CLUSTER_INFO[d["cluster"]]
    st.markdown(f"""
    <div style="margin:0 80px 32px;padding:28px;background:{cl['bg']};
         border:1px solid {cl['color']}44;border-radius:16px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
        <span style="font-size:28px">{cl['icon']}</span>
        <div>
          <div style="font-size:22px;font-weight:800;">{sel_gu}</div>
          <div style="font-size:13px;color:{cl['color']};font-weight:600;">{cl['name']}</div>
        </div>
      </div>
      <div class="metric-row">
        <div class="metric-card"><div class="metric-val" style="color:{cl['color']}">{d['risk']:.2f}</div><div class="metric-label">위험지수</div></div>
        <div class="metric-card"><div class="metric-val">{d['accident']}</div><div class="metric-label">총 사고건수</div></div>
        <div class="metric-card"><div class="metric-val">{d['epdo']}</div><div class="metric-label">EPDO 점수</div></div>
        <div class="metric-card"><div class="metric-val">{d['usage']:,}</div><div class="metric-label">이용건수(평균)</div></div>
        <div class="metric-card"><div class="metric-val">{d['intersection_ratio']*100:.0f}%</div><div class="metric-label">교차로 사고 비율</div></div>
        <div class="metric-card"><div class="metric-val">{d['shared_ratio']*100:.0f}%</div><div class="metric-label">겸용도로 비율</div></div>
      </div>
      <div style="font-size:13px;color:#aaa;">{cl['desc']}</div>
      <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
        {''.join(f'<span style="background:{cl["color"]}22;color:{cl["color"]};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;">✓ {p}</span>' for p in cl["policy"])}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Q1 — 어디가 얼마나 위험한가?
# ═══════════════════════════════════════════════════════════════════════════
if layer == "Q1":
    st.markdown("""
    <div class="section">
      <div class="section-header">
        <div class="section-num">01</div>
        <div>
          <div class="section-title">서울에서 자전거를 탄다면, 어디가 얼마나 위험한가?</div>
          <div class="section-sub">EPDO 기반 이용량 보정 위험지수 산출 및 구별 비교</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # H1 배너
    st.markdown("""
    <div style="margin:0 80px 16px;">
      <div style="background:rgba(76,201,240,0.08);border:1px solid rgba(76,201,240,0.2);border-radius:10px;padding:16px 20px;">
        <div style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:2px;color:#4cc9f0;margin-bottom:4px;">H1 — 가설</div>
        <div style="font-size:14px;font-weight:600;">이용량 보정 위험도(EPDO/이용량×10,000)는 단순 사고건수와 다른 결과를 보인다</div>
        <div style="font-size:12px;color:#888;margin-top:8px;">EPDO = 사망자수×3 + 중상자수×2 + 경상자수×1 + 부상신고자수×0.5 &nbsp;|&nbsp; 위험지수 = EPDO ÷ 이용건수 × 10,000</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_q1 = (pd.DataFrame([{"자치구": k, **v} for k, v in GU_DATA.items()])
             .sort_values("risk", ascending=False).reset_index(drop=True))

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        bar_colors = ["#e63946" if g in DANGER_TOP5 else ("#43aa8b" if g in SAFE_TOP5 else "#3a3a5c")
                      for g in df_q1["자치구"]]
        fig_bar = go.Figure(go.Bar(
            y=df_q1["자치구"], x=df_q1["risk"], orientation="h",
            marker_color=bar_colors,
            text=df_q1["risk"].round(2), textposition="outside",
            textfont=dict(size=11, color="white"),
        ))
        fig_bar.update_layout(
            title=dict(text="자치구별 이용량 보정 위험지수", font=dict(size=14, color="white")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Noto Sans KR"),
            height=700, margin=dict(l=10, r=80, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    with col_table:
        df_rank = df_q1[["자치구", "risk", "accident"]].copy()
        df_rank["위험지수_순위"] = range(1, len(df_rank) + 1)
        df_acc_s = df_q1.sort_values("accident", ascending=False).reset_index(drop=True)
        df_acc_s["사고건수_순위"] = range(1, len(df_acc_s) + 1)
        df_m = df_rank.merge(df_acc_s[["자치구", "사고건수_순위"]], on="자치구")
        df_m["순위변동"] = df_m["사고건수_순위"] - df_m["위험지수_순위"]

        rows_html = ""
        for _, r in df_m.iterrows():
            d_val = r["순위변동"]
            arrow = (f"<span style='color:#43aa8b'>▲{int(d_val)}</span>" if d_val > 0 else
                     (f"<span style='color:#e63946'>▼{int(-d_val)}</span>" if d_val < 0 else "–"))
            badge = ('<span class="badge-danger">위험</span>' if r["자치구"] in DANGER_TOP5 else
                     ('<span class="badge-safe">안전</span>' if r["자치구"] in SAFE_TOP5 else ""))
            rows_html += f"""<tr>
              <td style="font-weight:600">{r['자치구']} {badge}</td>
              <td style="text-align:center;color:#ff6b7a;font-weight:700">{int(r['위험지수_순위'])}</td>
              <td style="text-align:center;color:#aaa">{int(r['사고건수_순위'])}</td>
              <td style="text-align:center">{arrow}</td>
            </tr>"""

        st.markdown(f"""
        <div style="font-size:13px;font-weight:600;margin-bottom:12px;color:#aaa;">📊 위험지수 vs 사고건수 순위 비교</div>
        <div style="background:#13131f;border:1px solid rgba(255,255,255,0.08);border-radius:12px;overflow:auto;max-height:650px;">
          <table class="rank-table">
            <thead><tr>
              <th>자치구</th><th style="text-align:center">위험지수</th>
              <th style="text-align:center">사고건수</th><th style="text-align:center">변동</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:32px 80px'>",
                unsafe_allow_html=True)

    # H2 배너
    st.markdown("""
    <div style="margin:0 80px 16px;">
      <div style="background:rgba(248,150,30,0.08);border:1px solid rgba(248,150,30,0.2);border-radius:10px;padding:16px 20px;">
        <div style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:2px;color:#f8961e;margin-bottom:4px;">H2 — 가설</div>
        <div style="font-size:14px;font-weight:600;">위험구와 안전구는 도로 유형 비율에서 유의한 차이를 보인다</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns(3)

    def radar(group_gus, title, color_hex):
        cats  = ["전용도로", "우선도로", "겸용도로", "교차로비율×100", "위험지수/2"]
        avail = [g for g in group_gus if g in GU_DATA]
        if not avail:
            return go.Figure()
        vals = [
            np.mean([GU_DATA[g]["road_exclusive"] for g in avail]),
            np.mean([GU_DATA[g]["road_priority"]  for g in avail]),
            np.mean([GU_DATA[g]["road_shared"]     for g in avail]),
            np.mean([GU_DATA[g]["intersection_ratio"] * 100 for g in avail]),
            np.mean([GU_DATA[g]["risk"] for g in avail]) / 2,
        ]
        # fillcolor: hex_to_rgba로 안전하게 변환
        fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
            fillcolor=hex_to_rgba(color_hex, 0.2),
            line=dict(color=color_hex, width=2), name=title,
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
        st.plotly_chart(radar(DANGER_TOP5, "🔴 위험구 Top5", "#e63946"),
                        use_container_width=True, config={"displayModeBar": False})
    with col_r2:
        st.plotly_chart(radar(SAFE_TOP5,   "🟢 안전구 Top5", "#43aa8b"),
                        use_container_width=True, config={"displayModeBar": False})
    with col_r3:
        keys   = ["road_exclusive", "road_priority", "road_shared"]
        labels = ["전용도로", "우선도로", "겸용도로"]
        dv = [np.mean([GU_DATA[g][k] for g in DANGER_TOP5 if g in GU_DATA]) for k in keys]
        sv = [np.mean([GU_DATA[g][k] for g in SAFE_TOP5   if g in GU_DATA]) for k in keys]
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(name="위험구", x=labels, y=dv, marker_color="#e63946", opacity=0.85))
        fig_diff.add_trace(go.Bar(name="안전구", x=labels, y=sv, marker_color="#43aa8b", opacity=0.85))
        fig_diff.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Noto Sans KR"),
            title=dict(text="도로유형 비율 비교 (%)", font=dict(size=13, color="white")),
            height=300, margin=dict(l=10, r=10, t=50, b=30),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_diff, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Q2 — 왜 위험 수준이 다른가?
#  · 군집 산점도 (밝은 테마 + 신뢰 타원 + 군집 중심 X)
#  · 박스플롯 제거
# ═══════════════════════════════════════════════════════════════════════════
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

    df2 = pd.DataFrame([{"자치구": k, **v} for k, v in GU_DATA.items()])

    col_sc, col_info = st.columns([3, 2])

    with col_sc:
        fig_sc = go.Figure()

        for c_id in [0, 1, 2]:
            sub    = df2[df2["cluster"] == c_id]
            x_vals = (sub["intersection_ratio"] * 100).values
            y_vals = (sub["shared_ratio"] * 100).values
            color  = C_COLOR[c_id]
            label  = C_LABEL[c_id]

            # ① 신뢰 타원
            if len(x_vals) >= 3:
                ex, ey = get_ellipse_points(x_vals, y_vals, n_std=1.5)
                ex = np.append(ex, ex[0])
                ey = np.append(ey, ey[0])
                fig_sc.add_trace(go.Scatter(
                    x=ex, y=ey, mode="lines", fill="toself",
                    fillcolor=hex_to_rgba(color, 0.12),      # ← 안전한 rgba 변환
                    line=dict(color=color, width=1.5, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ))

            # ② 산점도
            fig_sc.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers+text",
                name=label,
                marker=dict(size=12, color=color,
                            line=dict(width=1, color="white"), opacity=0.95),
                text=sub["자치구"].values,
                textposition="top center",
                textfont=dict(size=10, color="#111", family="Noto Sans KR"),
            ))

            # ③ 군집 중심 X
            if len(x_vals) > 0:
                fig_sc.add_trace(go.Scatter(
                    x=[x_vals.mean()], y=[y_vals.mean()],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color=color,
                                line=dict(width=2.5, color=color)),
                    name="군집 중심" if c_id == 0 else None,
                    showlegend=(c_id == 0),
                    hovertemplate=f"군집 중심<br>교차로: {x_vals.mean():.1f}%<br>겸용도로: {y_vals.mean():.1f}%<extra></extra>",
                ))

        fig_sc.update_layout(
            title=dict(
                text="<b>서울시 자치구별 K-means 군집화 (k=3)</b>",
                font=dict(size=17, color="#111"), x=0.5, xanchor="center"
            ),
            xaxis=dict(
                title="<b>교차로 사고 비율 (%)</b><br><sub>(전체 자전거 사고 중 교차로 발생 비율)</sub>",
                gridcolor="#e9ecef", color="#333",
                range=[-2, 105], zeroline=False,
                showline=True, linewidth=1, linecolor="#bbb", mirror=True,
            ),
            yaxis=dict(
                title="<b>겸용도로 비율 (%)</b><br><sub>(전체 자전거도로 중 겸용도로 비율)</sub>",
                gridcolor="#e9ecef", color="#333",
                range=[-2, 105], zeroline=False,
                showline=True, linewidth=1, linecolor="#bbb", mirror=True,
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#333", family="Noto Sans KR"),
            height=640,
            margin=dict(l=60, r=30, t=70, b=70),
            legend=dict(
                x=0.01, y=0.01,
                bgcolor="white", bordercolor="#ddd", borderwidth=1,
                font=dict(size=11, color="#333"),
            ),
        )

        # 이미지의 보라색 테두리 연출
        st.markdown('<div style="border:3px solid #7b2d8b;border-radius:6px;overflow:hidden;background:white;">',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_sc, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # 범례 설명 (이미지 하단 텍스트 스타일)
        st.markdown("""
        <div style="background:white;border:1px solid #ddd;border-radius:8px;
             padding:12px 20px;margin-top:8px;font-size:12px;color:#444;">
          <b>군집 특성 요약</b><br>
          🔴 <b>군집 3: 교차로 집중 위험형</b> — 교차로 사고 비율 높음 / 겸용도로 낮음<br>
          🟡 <b>군집 2: 혼합 위험형</b> — 교차로 사고 비율 중간 / 겸용도로 비율 높음<br>
          🟢 <b>군집 1: 저위험 인프라형</b> — 교차로 사고 비율 낮음 / 겸용도로 낮음<br>
          <span style="color:#888;font-size:11px;">주: 교차로 사고 비율(23~24 교차로 자전거 사고/전체 자전거 사고) × 겸용도로비율(겸용도로/자전거도로 계)</span>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        for c_id in [2, 1, 0]:          # 저위험 → 혼합 → 고위험 순
            c_info = CLUSTER_INFO[c_id]
            gu_list = ", ".join(c_info["gu"][:5]) + ("..." if len(c_info["gu"]) > 5 else "")
            st.markdown(f"""
            <div style="background:{c_info['bg']};border:1px solid {c_info['color']}44;
                 border-radius:14px;padding:20px;margin-bottom:14px;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span style="font-size:24px">{c_info['icon']}</span>
                <div>
                  <div style="font-size:15px;font-weight:700;">{c_info['name']}</div>
                  <div style="font-size:11px;color:{c_info['color']}">Cluster {c_id}</div>
                </div>
              </div>
              <div style="font-size:12px;color:#bbb;line-height:1.6;margin-bottom:10px;">{c_info['desc']}</div>
              <div style="font-size:11px;color:#888;">대표 자치구: {gu_list}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Q3 — 무엇을 해야 하는가?
# ═══════════════════════════════════════════════════════════════════════════
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
                f'<span style="background:{c_info["color"]}22;color:{c_info["color"]};'
                f'padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;'
                f'margin:3px 3px 0 0;display:inline-block;">{g}</span>'
                for g in c_info["gu"]
            ])
            st.markdown(f"""
            <div style="background:{c_info['bg']};border:1px solid {c_info['color']}44;
                 border-radius:16px;padding:28px;min-height:420px;">
              <div style="font-size:36px;margin-bottom:12px;">{c_info['icon']}</div>
              <div style="font-size:18px;font-weight:800;margin-bottom:6px;">{c_info['name']}</div>
              <div style="font-size:12px;color:#aaa;margin-bottom:20px;line-height:1.6;">{c_info['desc']}</div>
              <div style="font-size:11px;font-weight:700;color:{c_info['color']};letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">권고 정책</div>
              {policy_items}
              <div style="margin-top:20px;font-size:11px;color:#888;margin-bottom:8px;">대상 자치구</div>
              <div>{gu_badges}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.08);margin:32px 80px'>",
                unsafe_allow_html=True)

    st.markdown('<div style="margin:0 80px 16px;font-size:15px;font-weight:700;">정책 우선순위 매트릭스 — 위험지수 × 군집 유형</div>',
                unsafe_allow_html=True)

    df3 = pd.DataFrame([{"자치구": k, **v} for k, v in GU_DATA.items()])
    cluster_names_q3 = {0: "교차로 집중형", 1: "혼합 위험형", 2: "저위험 인프라형"}

    fig_pri = go.Figure()
    for c_id in [0, 1, 2]:
        sub = df3[df3["cluster"] == c_id].sort_values("risk", ascending=False)
        fig_pri.add_trace(go.Bar(
            x=sub["자치구"], y=sub["risk"],
            name=f"{CLUSTER_INFO[c_id]['icon']} {cluster_names_q3[c_id]}",
            marker_color=C_COLOR[c_id], opacity=0.85,
            text=sub["risk"].round(1), textposition="outside",
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

    st.markdown("""
    <div style="margin:20px 80px 60px;padding:28px 36px;
         background:linear-gradient(135deg,rgba(255,77,109,0.1) 0%,rgba(76,201,240,0.06) 100%);
         border:1px solid rgba(255,77,109,0.25);border-radius:16px;">
      <div style="font-size:18px;font-weight:800;margin-bottom:12px;">🎯 핵심 결론</div>
      <div style="font-size:14px;color:#ccc;line-height:1.8;">
        서울시 자치구의 자전거 사고 위험은 <strong style="color:#ff4d6d">이용량 보정 후 사고건수 순위와 크게 달라지며</strong>,
        그 원인은 구마다 상이한 <strong style="color:#4cc9f0">도로 환경 구조</strong>에서 비롯됩니다.<br><br>
        K=3 군집 분석 결과, 자치구는 ①교차로 집중형 ②혼합 위험형 ③저위험 인프라형으로 분류되며,
        각 유형에 맞는 <strong style="color:#f8961e">맞춤형 정책</strong>이 요구됩니다.
        <strong>원인에 기반한 구별 차별화 전략</strong>이 서울시 자전거 안전의 핵심입니다.
      </div>
    </div>
    """, unsafe_allow_html=True)
