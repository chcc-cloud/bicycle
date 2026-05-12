import streamlit as st
import pandas as pd
import sqlite3
import os
import plotly.express as px

# 페이지 기본 설정 (가장 먼저 와야 함)
st.set_page_config(page_title="공공데이터 경제 분석 대시보드", layout="wide")

# 🎈 제목
st.title("📊 대한민국 경제 현황 대시보드")
st.markdown("데이터 초보자도 쉽게 보는 경제 지표 분석입니다. 각 차트 아래에서 **사용된 SQL 코드**도 확인할 수 있어요!")

# 🛑 데이터베이스 파일 확인 (요청하신 친절한 에러 메시지)
db_path = "economics.db"
if not os.path.exists(db_path):
    st.error("앗! 데이터베이스 파일을 찾을 수 없어요. 😢")
    st.info("💡 해결 방법: 'app.py' 파일이 있는 폴더에 'economics.db' 파일이 함께 있는지 확인해주세요!")
    st.stop() # 파일이 없으면 여기서 멈춤

# 🔌 데이터베이스 연결 함수
@st.cache_data # 데이터를 매번 다시 부르지 않게 기억(캐시)해두는 마법의 주문
def load_data(query):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# 차트와 SQL을 예쁘게 보여주는 도우미 함수
def display_chart_and_sql(title, question, chart_fig, sql_query):
    st.subheader(title)
    st.markdown(f"**궁금증:** {question}")
    st.plotly_chart(chart_fig, use_container_width=True)
    with st.expander("🔍 사용된 SQL 코드 보기"):
        st.code(sql_query, language="sql")
    st.divider()

# =======================================================
# 📈 1. 성별 취업·실업 비교 (묶음 막대 차트)
# =======================================================
sql_1 = """
SELECT 성별, SUM(취업자) AS 취업자, SUM(실업자) AS 실업자 
FROM 경제활동현황 
WHERE 성별 != '계' -- 전체 합계 데이터가 있다면 제외
GROUP BY 성별
"""
df_1 = load_data(sql_1)
# Plotly를 이용해 보기 좋게 데이터를 변형(Melt)해서 그립니다.
df_1_melted = df_1.melt(id_vars='성별', value_vars=['취업자', '실업자'], var_name='상태', value_name='인원수')
fig_1 = px.bar(df_1_melted, x='성별', y='인원수', color='상태', barmode='group', title="성별 취업자/실업자 비교")
display_chart_and_sql("1. 성별 취업·실업 비교", "남녀 간 취업·실업 격차는 얼마나 될까?", fig_1, sql_1)


# =======================================================
# 📈 2. 실업률 시계열 추이 (꺾은선 차트)
# =======================================================
sql_2 = """
SELECT 시점, AVG(CAST(실업률 AS REAL)) AS 평균실업률 
FROM 경제활동현황 
GROUP BY 시점 
ORDER BY 시점
"""
df_2 = load_data(sql_2)
fig_2 = px.line(df_2, x='시점', y='평균실업률', markers=True, title="연도/월별 평균 실업률 추이")
display_chart_and_sql("2. 실업률 시계열 추이", "시점별 실업률은 어떻게 변화했을까?", fig_2, sql_2)


# =======================================================
# 📈 3. 산업별 취업자 TOP 10 (가로 막대 차트)
# =======================================================
sql_3 = """
SELECT 산업별, SUM(취업자수) AS 총취업자수 
FROM 산업별_취업자 
GROUP BY 산업별 
ORDER BY 총취업자수 DESC 
LIMIT 10
"""
df_3 = load_data(sql_3)
fig_3 = px.bar(df_3, x='총취업자수', y='산업별', orientation='h', title="취업자가 가장 많은 산업 TOP 10")
fig_3.update_layout(yaxis={'categoryorder': 'total ascending'}) # 1등이 맨 위로 오게 뒤집기
display_chart_and_sql("3. 산업별 취업자 TOP 10", "어떤 산업에 취업자가 가장 많을까?", fig_3, sql_3)


# =======================================================
# 📈 4. 산업별 임금 vs 근로시간 (산점도/버블 차트)
# =======================================================
sql_4 = """
SELECT 산업별, AVG(전체근로시간) AS 평균근로시간, AVG(전체임금총액) AS 평균임금 
FROM 산업별_임금_및_근로시간 
GROUP BY 산업별
"""
df_4 = load_data(sql_4)
fig_4 = px.scatter(df_4, x='평균근로시간', y='평균임금', text='산업별', size='평균임금', 
                   color='산업별', title="산업별 평균 임금과 근로시간의 관계")
fig_4.update_traces(textposition='top center') # 글자가 점 위에 오도록 설정
display_chart_and_sql("4. 산업별 임금 vs 근로시간", "임금이 높은 산업이 근로시간도 길까?", fig_4, sql_4)


# =======================================================
# 📈 5. 비경제활동 이유 분포 (도넛 차트)
# =======================================================
sql_5 = """
SELECT 
    SUM("몸이 좋지 않아 쉬고 있음") AS "건강 문제",
    SUM("퇴사(정년 퇴직)후 계속 쉬고 있음") AS "퇴사/정년",
    SUM("일의 완료, 고용계약이 만료되어 쉬고 있음") AS "계약 만료",
    SUM(CAST("직장의 휴업·폐업으로 쉬고 있음" AS INTEGER)) AS "휴업/폐업",
    SUM("원하는 일자리(일거리)를 찾기 어려워 쉬고 있음") AS "구직 난항",
    SUM("일자리(일거리)가 없어서 쉬고 있음") AS "일자리 없음",
    SUM("다음 일 준비를 위해 쉬고 있음") AS "다음 일 준비",
    SUM("기타") AS "기타"
FROM 쉬었음의_주된_이유
"""
df_5 = load_data(sql_5)
# 도넛 차트를 그리기 위해 데이터를 세로로 길게 변환(Melt)
df_5_melted = df_5.melt(var_name='이유', value_name='인원수')
fig_5 = px.pie(df_5_melted, names='이유', values='인원수', hole=0.4, title="사람들이 쉬고 있는 주된 이유")
fig_5.update_traces(textposition='inside', textinfo='percent+label')
display_chart_and_sql("5. 비경제활동 이유 분포", "사람들이 쉬고 있는 가장 큰 이유는?", fig_5, sql_5)


# =======================================================
# 📈 6. 경제활동 참가율 추이 (이중 꺾은선 차트)
# =======================================================
# 두 테이블을 '시점'을 기준으로 연결(JOIN)합니다.
sql_6 = """
SELECT A.시점, A.총경제활동인구, B.비경제활동인구
FROM (
    SELECT 시점, SUM(경제활동인구) AS 총경제활동인구 
    FROM 경제활동현황 
    GROUP BY 시점
) A
JOIN 비경제활동인구 B ON A.시점 = B.시점
ORDER BY A.시점
"""
df_6 = load_data(sql_6)
# 두 개의 선을 함께 그리기 위해 데이터 변환
df_6_melted = df_6.melt(id_vars='시점', value_vars=['총경제활동인구', '비경제활동인구'], 
                        var_name='구분', value_name='인구수')
fig_6 = px.line(df_6_melted, x='시점', y='인구수', color='구분', markers=True, 
                title="경제활동인구 vs 비경제활동인구 추이")
display_chart_and_sql("6. 경제활동 참가율 추이", "비경제활동인구는 시간이 갈수록 늘고 있나?", fig_6, sql_6)

st.success("🎉 모든 데이터 분석이 완료되었습니다! 고생하셨습니다.")