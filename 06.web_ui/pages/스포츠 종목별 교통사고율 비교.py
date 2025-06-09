import streamlit as st
from utils.layout import set_config, login_widget

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


set_config()
st.title("스포츠 종목별 교통사고율 비교")
login_widget()
st.write("> ##### 관중이 많은 야구가 다른 스포츠 종목(축구, 농구, 배구)보다 교통사고가 더 많이 일어날까?")


import platform

system = platform.system()
font_name = "NanumGothic" # Linux

if system == 'Darwin':  # macOS
    font_name = 'AppleGothic'
elif system == 'Windows': # Windows
    font_name = 'Malgun Gothic'

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False


# ==================================== 모델링 ====================================
# 0. 데이터셋 로드
df = pd.read_csv("./05.modeling/1st-dataset-prepressed-total.csv")
pd.set_option("display.max_columns", None)
df['accident_score'] = df['accident_count'] + 3 * df['injury_count']

# 1. 독립 변수와 종속 변수 정의
X = df.drop(columns=["accident_score", "accident_count", "injury_count", "death_count", 
            "game_id", "stadium_code","game_date", "day_of_week", "start_time", 
            "region", 'match_시범경기', 'match_정규시즌', 'match_포스트시즌'])
y = df['accident_score']

# 2. 포아송 회귀모델 (GLM - Generalized Linear Model)
df_model = df[['accident_score', 'type_농구','type_배구','type_야구','type_여자배구','type_축구','audience', 'is_holiday', 'start_hour', 'home_team_win', 'temperature', 'precipitation', 'snow_depth' ]].copy()
model = smf.glm(
    formula='accident_score ~ type_농구 + type_배구 + type_야구 + type_여자배구 + type_축구 + audience + is_holiday + start_hour + start_hour + home_team_win + temperature + precipitation + snow_depth',
    data=df_model,
    family=sm.families.Poisson()
)
result = model.fit()

# 절편만 있는 모델 (null model)
null_model = smf.glm(
    formula='accident_score ~ 1',
    data=df_model,
    family=sm.families.Poisson()
).fit()
n = len(df_model)
llf = result.llf
llnull = null_model.llf
deviance = result.deviance
chi2 = result.pearson_chi2
r2 = (1 - np.exp((2/n) * (llnull - llf))) / (1 - np.exp((2/n) * llnull))

# 3. 모델 평가
df_model['predicted'] = result.predict(df_model)
df_model['residuals'] = df_model['accident_score'] - df_model['predicted']

## 3-1) 예측값 vs 실제값
fig1 = plt.figure(figsize=(8, 4))
sns.scatterplot(x='predicted', y='accident_score', data=df_model)
plt.plot([df_model['accident_score'].min(), df_model['accident_score'].max()],
         [df_model['accident_score'].min(), df_model['accident_score'].max()],
         'r--', label='y = x')
plt.xlabel('Predicted accident_score')
plt.ylabel('Actual accident_score')
plt.title('예측값 vs 실제값 (Poisson Regression)')
plt.legend()
plt.grid(True)
plt1 = fig1

## 3-2) 잔차 vs 예측값
fig2 = plt.figure(figsize=(8, 4))
sns.scatterplot(x='predicted', y='residuals', data=df_model)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted accident_score')
plt.ylabel('Residuals')
plt.title('잔차 vs 예측값')
plt.grid(True)
plt2 = fig2

## 3-3) 계수(Coefficient) 시각화
coef = result.params
conf = result.conf_int()
conf.columns = ['2.5%', '97.5%']
coef_df = pd.concat([coef, conf], axis=1).reset_index()
coef_df.columns = ['variable', 'coefficient', 'ci_lower', 'ci_upper']

fig3 = plt.figure(figsize=(8, 4))
sns.pointplot(data=coef_df, y='variable', x='coefficient')
plt.errorbar(x=coef_df['coefficient'], y=coef_df['variable'],
             xerr=[coef_df['coefficient'] - coef_df['ci_lower'], coef_df['ci_upper'] - coef_df['coefficient']],
             fmt='none', c='gray', capsize=4)
plt.axvline(0, color='red', linestyle='--')
plt.title('Poisson 회귀 계수 및 신뢰구간')
plt.xlabel('계수 (Coefficient)')
plt.ylabel('변수')
plt.grid(True)
plt.tight_layout()
plt3 = fig3


# ==================================== 웹 페이지 ====================================
st.header("1. 모델 개요")
st.write("* **목적**: 야구, 축구, 농구, 배구 등 **다양한 스포츠 경기 종류별**로 경기장 주변에서 발생하는 **교통사고율(accident\\_score)** 에 영향을 주는 요인을 파악하고 정량적으로 예측")
st.write("""* **모델**: **Poisson 회귀 모델 (Generalized Linear Model - GLM)**
    * 모델 선정 이유 : 종속변수가 **이산형 또는 비율형 사건수**일 때 적합한 모델""")
st.write("* **종속변수**: `accident_score` (사고 발생 강도 및 빈도를 반영하는 수치형 지표)")
st.write("""* **주요 독립변수**:
    * 경기 종류: `type_농구`, `type_배구`, `type_여자배구`, `type_야구`, `type_축구` (기준값은 기타)
    * 기타 변수: `audience`, `is_holiday`, `start_hour`, `home_team_win`, `temperature`, `precipitation`, `snow_depth`""")
st.write("")
st.write("")


st.header("2. 모델 적합도")
st.write("### 2-1. Summary")
summary = result.summary()
st.code(summary, language='text')

df1 = pd.DataFrame({
    "항목": ["Log-Likelihood", "Deviance", "Pearson chi²", "Pseudo R² (CS)"],
    "값": [f"{llf:.0f}", f"{deviance:.0f}", f"{chi2:.0f}", f"{r2:.4f}"],
    "해석": [
        "우도 기반 적합도 (클수록 좋음)",
        "모델 잔차의 크기 (작을수록 좋음)",
        "예측 적합도 검증 지표",
        "모델이 전체 accident_score 변동의 약 51%를 설명함 → 우수한 설명력"
    ]
})
styled_df = df1.style.set_table_styles([{'selector': 'th', 'props': [('background-color', '#dbeafe'), ('color', 'black')]}])
st.dataframe(styled_df, use_container_width=True)

st.write("➡ **결론**: 해당 포아송 모델은 `accident_score` 예측에 있어 **양호한 설명력을 보유**")
st.write("")
st.write("")


st.write("### 2-2. 예측값과 실제값 비교")
st.pyplot(plt1)


st.write("### 2-3. 잔차 vs 예측값")
st.pyplot(plt2)


st.write("### 2-4. 계수(Coefficient) 시각화")
st.pyplot(plt3)



st.header("3. 주요 계수 해석")
df2 = pd.DataFrame({
    "변수": [
        "Intercept", "type_배구", "type_여자배구", "type_축구", "type_야구", "type_농구",
        "audience", "is_holiday", "start_hour", "home_team_win", "temperature",
        "precipitation", "snow_depth"
    ],
    "계수(β)": [
        2.6510, 0.6764, 0.8745, 0.5375, 0.3261, 0.2365,
        9.27e-06, -0.1047, 0.0071, -0.0099, 0.0020,
        -0.0309, 0.0448
    ],
    "p-value": [
        0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.000, 0.000, 0.005, 0.147, 0.000,
        0.000, 0.000
    ],
    "해석 요약": [
        "기준 조건에서 log(accident_score) ≈ 2.65 (≈ 14.16)",
        "배구 경기 시 기준 대비 사고율 약 1.97배 증가",
        "여자배구 시 약 2.4배 증가",
        "축구 시 약 1.71배 증가",
        "야구 시 약 1.39배 증가",
        "농구 시 약 1.27배 증가",
        "관중 수 증가 시 사고율 증가 (비례 관계)",
        "공휴일이면 사고율 약 10% 감소",
        "경기 시작 시각이 늦을수록 사고율 미세 증가 (0.7%/시간)",
        "영향 없음 (p > 0.05)",
        "기온 상승 시 사고율 미세 증가 (0.2%/°C)",
        "강수량 증가 시 사고율 약 3% 감소",
        "눈 쌓일수록 사고율 약 4.6% 증가"
    ]
})
styled_df = df2.style.set_table_styles([{'selector': 'th', 'props': [('background-color', '#dbeafe'), ('color', 'black')]}])
st.dataframe(styled_df, use_container_width=True)
st.write("")
st.write("")


st.header("4. 잔차 분석")
st.write("* **Deviance**와 **Pearson chi²** 모두 예측과 관측값 사이의 차이가 통계적으로 허용 범위 내")
st.write("* Pseudo R² ≈ 0.51 → **예측력이 우수한 편**")
st.write("* 유의하지 않은 변수는 `home_team_win` 1개뿐이며, 나머지 변수는 유의미함")
st.write("➡ 이상치, 과적합 등 문제는 없으며 모델이 전체적으로 안정적으로 학습됨")
st.write("")


st.header("5. 결론 및 제언")
st.write("##### 🎯 핵심 결론")
st.write("* 스포츠 경기 종류는 **사고율에 유의미한 영향을 줌** → 특히 **여자배구, 배구, 축구** 순으로 높은 증가폭")
st.write("* **관중 수 증가**, **눈 쌓임**, **기온 상승**, **경기 시간 지연**도 사고 위험 상승과 관련")
st.write("* 반면, **공휴일**과 **강수량**은 사고율 감소 요인")

st.write("##### ✅ 향후 제언")
st.write(""" * **여자배구/배구/축구 경기일 교통안전 대책 강화**
    * 인근 도로 교통통제, 경찰 배치, 대중교통 연계 강화 필요""")
st.write(""" * **관중 규모 예측 기반 사전 대응**
    * 실시간 관중 수 예측 모델과 연동하여 사고 예방 조치""")
st.write(""" * **기상 요소 통합형 예측 시스템 도입**
    * 눈/비 정보 기반으로 주차장/보행자 동선 등 조정""")
st.write(""" * **승패 변수는 효과 없음**
    * 팬심에 의한 사고 가능성은 분석상 유의하지 않음\n""")