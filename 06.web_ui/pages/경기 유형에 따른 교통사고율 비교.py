import streamlit as st
import pandas as pd
from utils.layout import set_config, login_widget
from PIL import Image

set_config()
st.title("경기 유형에 따른 교통사고율 비교")
login_widget()

st.write("> ##### 일반 경기보다 특별 이벤트 경기(포스트시즌 등)가 있는 날에 교통사고가 더 많이 일어날까?")


st.header("1. 모델 개요")
st.write("* **목적**: 야구 경기 관련 요인이 **사고 위험 점수(`accident_score`)** 에 유의미한 영향을 주는지 정량적으로 평가")
st.write("* **모델**: 포아송 회귀 (Generalized Linear Model, `family=Poisson`, `link=log`)")
st.write("* **종속변수**: `accident_score` (사고 수와 심각도 반영한 지표로 추정)")
st.write("""* **독립변수**:
    * 경기 유형: `match_시범경기`, `match_정규시즌`, `match_포스트시즌`, `audience`, `home_team_win`
    * 시간 및 환경 변수: `start_hour`, `temperature`, `precipitation`
    * 기타 영향 요인: `is_holiday`""")
st.write("")
st.write("")


st.header("2. 모델 적합도")
summary = """                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:         accident_score   No. Observations:                 2072
Model:                            GLM   Df Residuals:                     2063
Model Family:                 Poisson   Df Model:                            8
Link Function:                    Log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -27702.
Date:                Thu, 05 Jun 2025   Deviance:                       47227.
Time:                        08:37:44   Pearson chi2:                 4.64e+04
No. Iterations:                     5   Pseudo R-squ. (CS):             0.4913
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept         1.7964      0.055     32.903      0.000       1.689       1.903
match_시범경기     0.4358      0.018     23.836      0.000       0.400       0.472
match_정규시즌     0.5127      0.026     19.694      0.000       0.462       0.564
match_포스트시즌    0.8479      0.031     27.752      0.000       0.788       0.908
is_holiday       -0.1232      0.012     -9.867      0.000      -0.148      -0.099
start_hour        0.0421      0.004      9.684      0.000       0.034       0.051
audience       2.446e-05   8.55e-07     28.612      0.000    2.28e-05    2.61e-05
home_team_win     0.0208      0.009      2.375      0.018       0.004       0.038
temperature      -0.0043      0.001     -5.245      0.000      -0.006      -0.003
precipitation    -0.0255      0.003     -7.314      0.000      -0.032      -0.019
================================================================================="""
st.code(summary, language='text')

df1 = pd.DataFrame({
    "항목": ["Log-Likelihood", "Deviance", "Pearson chi²", "Pseudo R² (CS)"],
    "값": ["-27702", "47227", "4.64e+04", "0.4913"],
    "해석": [
        "모델의 우도 기반 적합도 지표",
        "모델 잔차 제곱합",
        "데이터의 분산과 잔차 사이의 차이 평가",
        "전체 설명력 약 49.13%로, 선형회귀 대비 대폭 향상됨"
    ]
})
styled_df = df1.style.set_table_styles([{'selector': 'th', 'props': [('background-color', '#dbeafe'), ('color', 'black')]}])
st.dataframe(styled_df, use_container_width=True)

st.write("➡ **결론**: 해당 포아송 모델은 `accident_score` 예측에 있어 **양호한 설명력을 보유**")
st.write("")
st.write("")


st.header("3. 주요 계수 해석 (log-link 기반 → $$exp(β)$$ 해석 가능)")
df2 = pd.DataFrame({
    "변수명": [
        "Intercept", "match_시범경기", "match_정규시즌", "match_포스트시즌", "is_holiday",
        "start_hour", "audience", "home_team_win", "temperature", "precipitation"
    ],
    "계수 (β)": [
        "1.796", "0.436", "0.513", "0.847", "-0.123",
        "0.042", "2.45e-05", "0.0208", "-0.0043", "-0.0255"
    ],
    "P-value": ["0.000"] * 9 + ["0.000"],
    "해석 요약": [
        "기준 조건에서 사고점수 log값이 1.796 → exp(1.796) ≈ 6.03점",
        "시범경기는 기준 대비 exp(0.436) ≈ 1.55배 위험도 증가",
        "정규시즌 경기일 경우 약 1.67배 증가",
        "포스트시즌일 경우 약 2.33배 증가 (가장 큰 영향)",
        "공휴일은 사고점수 약 12% 감소 (exp(-0.123) ≈ 0.88)",
        "경기 시작 시간 1시간 증가 시 4.3% 증가",
        "관중 수 증가 → 위험도 비례 증가 (10,000명당 약 27.6% 증가)",
        "승리 시 약간 증가 (약 2.1%) – 미미하지만 유의",
        "기온 1도 상승 시 사고점수 약 0.4% 감소",
        "강수량 증가 시 사고점수 약 2.5% 감소"
    ]
})
styled_df = df2.style.set_table_styles([{'selector': 'th', 'props': [('background-color', '#dbeafe'), ('color', 'black')]}])
st.dataframe(styled_df, use_container_width=True)

st.write("☝🏻 포스트시즌 경기가 위험도를 가장 크게 증가시키는 요인")
st.write("✌🏻 공휴일, 기온, 강수는 위험도를 줄이는 요인")
st.write("👌🏻 관중 수가 높고, 경기 시작 시간이 늦을 수록 위험도가 상승")
st.write("")
st.write("")


st.header("4. 모델 특이점 및 비교")
df3 = pd.DataFrame({
    "항목": ["설명력 (R²)", "유의 변수 수", "해석 방식"],
    "선형회귀": ["3% 내외", "일부만", "절대 증가량"],
    "포아송회귀": [
        "49.1% (대폭 향상)",
        "모든 변수 통계적으로 유의",
        "비율(%) 기반 해석 가능 (exp(β))"
    ]
})
styled_df = df3.style.set_table_styles([{'selector': 'th', 'props': [('background-color', '#dbeafe'), ('color', 'black')]}])
st.dataframe(styled_df, use_container_width=True)

st.write("➡ **포아송 회귀는 `accident_score` 같은 이산적/비선형 지표 예측에 더 적합**")
st.write("")
st.write("")


st.header("5. 결론 및 제언")
st.write("##### 🎯 핵심 결론")
st.write("* 포스트시즌 경기, 관중 수, 경기 시작 시각이 **사고 위험도 증가 요인**")
st.write("* 공휴일, 기온, 강수는 **사고 위험도 감소 요인**")
st.write("* 모델의 설명력 및 예측 적합도는 **선형회귀에 비해 월등히 우수**")

st.write("##### ✅ 향후 제언")
st.write("* 모델 활용성 확대하여 사고 예방 캠페인, 교통 통제 계획, 관중 수 제한 정책 등에 활용 가능함")
st.write("* 데이터 확장을 통해 지역 기반 변수, 경기장 별 인프라, 요일 등 추가 시 더 높은 정확도 기대")
st.write("* 모델 정교화를 적용하여 Zero-Inflated Poisson (ZIP), Negative Binomial 등도 향후 적용 고려")
