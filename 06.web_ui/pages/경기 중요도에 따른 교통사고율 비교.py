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
st.title("경기 중요도에 따른 교통사고율 비교")
login_widget()
st.write("> ##### 일반 경기보다 특별 이벤트 경기(포스트시즌 등)가 있는 날에 교통사고가 더 많이 일어날까?")


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
# 0. 데이터셋 로드 + 결측치 처리
df = pd.read_csv('./05.modeling/1st-dataset-prepressed-baseball.csv')

pd.set_option("display.max_columns", None)
df['accident_score'] = df['accident_count'] + 3 * df['injury_count']
def pct(n): return f"{n:.1%}"


# 1. 독립 변수와 종속 변수 정의
X = df.drop(columns=["accident_score", "accident_count", "injury_count", "death_count", "game_id", "stadium_code","sports_type","game_date", "day_of_week", "start_time", "region", "snow_depth"])
y = df['accident_score']


# 2. 포아송 회귀모델 (GLM - Generalized Linear Model)
df_model = df[['accident_score', 'match_시범경기', 'match_정규시즌', 'match_포스트시즌', 'is_holiday', 'audience', 'start_hour', 'home_team_win', 'temperature', 'precipitation' ]].copy()
df_model['match_시범경기'] = df_model['match_시범경기'].astype(int)
df_model['match_정규시즌'] = df_model['match_정규시즌'].astype(int)
df_model['match_포스트시즌'] = df_model['match_포스트시즌'].astype(int)

model = smf.glm(
    formula='accident_score ~ match_시범경기 + match_정규시즌 + match_포스트시즌 + is_holiday + start_hour + audience + start_hour + home_team_win + temperature + precipitation',
    data=df_model,
    family=sm.families.Poisson()
)
result = model.fit()
summary = result.summary()


# 3. 모델 평가
## 3-1) 예측값 vs 실제값
df_model['predicted'] = result.predict(df_model)
df_model['residuals'] = df_model['accident_score'] - df_model['predicted']
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
st.write("* **목적**: 야구 경기 관련 요인이 **사고 위험 점수(`accident_score`)** 에 유의미한 영향을 주는지 정량적으로 평")
st.write("* **모델**: 포아송 회귀 (Generalized Linear Model, `family=Poisson`, `link=log`)")
st.write("* **종속변수**: `accident_score` (사고 수와 심각도 반영한 지표로 추정)")
st.write("""* **독립변수**:
    * 경기 유형: `match_시범경기`, `match_정규시즌`, `match_포스트시즌`, `audience`, `home_team_win`
    * 시간 및 환경 변수: `start_hour`, `temperature`, `precipitation`
    * 기타 영향 요인: `is_holiday`""")
st.write("")
st.write("")


st.header("2. 모델 적합도")
st.write("### 2-1. Summary")
summary = result.summary()
st.code(summary, language='text')
st.write("""| 항목                 | 값          | 해석                                           |
| ------------------ | ---------- | -------------------------------------------- |
| Log-Likelihood     | -27702     | 모델의 우도 기반 적합도 지표 (높을수록 좋음)                   |
| Deviance           | 47227      | 모델 잔차 제곱합 – 작을수록 적합도 우수                      |
| Pearson chi²       | 4.64e+04   | 데이터의 분산과 잔차 사이의 차이 평가                        |
| **Pseudo R² (CS)** | **0.4913** | 전체 설명력 약 \\*\\*49.13%\\*\\*로, 선형회귀 대비 **대폭 향상**됨 |
""")
st.write("➡ **결론**: 해당 포아송 모델은 `accident_score` 예측에 있어 **양호한 설명력을 보유**")

st.write("### 2-2. 예측값과 실제값 비교")
st.pyplot(plt1)

st.write("### 2-3. 잔차 vs 예측값")
st.pyplot(plt2)

st.write("### 2-4. 계수(Coefficient) 시각화")
st.pyplot(plt3)
st.write("")
st.write("")


st.header("3. 주요 계수 해석")
st.write("""| 변수명                 | 계수 (β)   | P-value | 해석 요약                                             |
| ------------------- | -------- | ------- | ------------------------------------------------- |
| **Intercept**       | 1.796    | 0.000   | 기준 조건에서 사고점수 log값이 1.796 → exp(1.796) ≈ **6.03점** |
| **match\\_시범경기**     | 0.436    | 0.000   | 시범경기는 기준 대비 **exp(0.436) ≈ 1.55배 위험도 증가**         |
| **match\\_정규시즌**     | 0.513    | 0.000   | 정규시즌 경기일 경우 **약 1.67배 증가**                        |
| **match\\_포스트시즌**    | 0.847    | 0.000   | 포스트시즌일 경우 **약 2.33배 증가** (가장 큰 영향)                |
| **is\\_holiday**     | -0.123   | 0.000   | 공휴일은 사고점수 **약 12% 감소** (exp(-0.123) ≈ 0.88)       |
| **start\\_hour**     | 0.042    | 0.000   | 경기 시작 시간 1시간 증가 시 **4.3% 증가**                     |
| **audience**        | 2.45e-05 | 0.000   | 관중 수 증가 → 위험도 비례 증가 (10,000명당 약 27.6% 증가)         |
| **home\\_team\\_win** | 0.0208   | 0.018   | 승리 시 약간 증가 (약 2.1%) – 미미하지만 유의                    |
| **temperature**     | -0.0043  | 0.000   | 기온 1도 상승 시 사고점수 약 0.4% 감소                         |
| **precipitation**   | -0.0255  | 0.000   | 강수량 증가 시 사고점수 약 2.5% 감소                           |
""")
st.write("➡ **핵심 해석**:")
st.write("* 포스트시즌이 가장 큰 위험도 증가 요인")
st.write("* 공휴일, 기온, 강수는 위험도를 줄이는 요인")
st.write("* 관중 수와 시작 시간이 높을수록 위험도 상승")
st.write("")
st.write("")


st.header("4. 모델 특이점 및 비교")
st.write("""| 항목       | 선형회귀   | 포아송회귀                       |
| -------- | ------ | --------------------------- |
| 설명력 (R²) | 3% 내외  | **49.1%** (대폭 향상)           |
| 유의 변수 수  | 일부만    | **모든 변수 통계적으로 유의**          |
| 해석 방식    | 절대 증가량 | **비율(%) 기반 해석** 가능 (exp(β)) |""")
st.write("➡ **포아송 회귀는 `accident_score` 같은 이산적/비선형 지표 예측에 더 적합**")
st.write("")
st.write("")


st.header("5. 결론 및 제언")
st.write("##### 🎯 핵심 결론")
st.write("""* 포스트시즌 경기, 관중 수, 경기 시작 시각이 **사고 위험도 증가 요인**
* 공휴일, 기온, 강수는 **사고 위험도 감소 요인**
* 모델의 설명력 및 예측 적합도는 **선형회귀에 비해 월등히 우수**""")

st.write("##### ✅ 향후 제언")
st.write("""1. **모델 활용성 확대**:
    * 사고 예방 캠페인, 교통 통제 계획, 관중 수 제한 정책 등에 활용 가능
2. **데이터 확장**:
    * 지역 기반 변수, 경기장 별 인프라, 요일 등 추가시 더 높은 정확도 기대
3. **모델 정교화**:
    * Zero-Inflated Poisson (ZIP), Negative Binomial 등도 향후 적용 고려""")