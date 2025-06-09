import streamlit as st
from utils.layout import set_config, login_widget

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import date, timedelta, time

set_config()
st.title("경기 유무에 따른 교통사고율 비교")
login_widget()
st.write("> ##### 스포츠 경기가 있는 날에는 교통사고가 더 많이 일어날까?")


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
df = pd.read_csv('./04.data_preprocess/2nd-dataset_20230101~20241231_전국_전종목_전체요일_20250605_113550.csv')

## 기존 데이터셋 결측치 처리
cols_to_fill = ["precipitation", "snow_depth", "temperature"]
df.loc[:, df.columns.intersection(cols_to_fill)] = (
    df.loc[:, df.columns.intersection(cols_to_fill)].fillna(0)
)

region_mean = df.groupby('region')['audience'].mean()
def impute_with_noise(row):
    if row['sports_type'] != '없음' and row['audience'] == 0:
        mean_val = region_mean.get(row['region'], np.nan)
        if pd.notna(mean_val):
            noise = np.random.normal(loc=0, scale=0.1)  # 평균 0, 표준편차 0.1 → ±10% 정도 오차
            return (mean_val * (1 + noise)).astype(int)
    return row['audience']
df['audience'] = df.apply(impute_with_noise, axis=1)
df['audience'] = df['audience'].astype(int)

## 모든 스포츠/요일 boolean 컬럼 생성
all_sports = set()
for sports_list in df['sports_type'].unique():
    if sports_list != "없음":
        for sport in sports_list.split(','):
            all_sports.add(sport.strip())
for sport in sorted(list(all_sports)):
    df[f'is_{sport}'] = df['sports_type'].apply(lambda x: sport in x)

## 모든 요일 boolean 컬럼 생성
all_weekday = set()
for day_list in df['weekday'].unique():
    for day in day_list:
        all_weekday.add(day.strip()) # .strip() to remove potential leading/trailing spaces
for day in sorted(list(all_weekday)):
    df[f'is_{day}'] = df['weekday'].apply(lambda x: day in x)
    
## 모든 날씨 더미 컬럼 생성
df = pd.get_dummies(df, columns=['weather_condition'])
df= df.drop('weather_condition_정보없음', axis=1)

## 사고 점수 가중치 부여    
df['accident_score'] = df['accident_count'] + 3 * df['injury_count']

df['date'] = pd.to_datetime(df['date'])
df['game_start_time'] = df['game_start_time'].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
df['game_start_time'] = pd.to_datetime(df['game_start_time'], format='%H:%M:%S', errors='coerce').dt.time
df['game_end_time'] = df['game_end_time'].str.extract(r'(\d{2}:\d{2}:\d{2})')[0]
df['game_end_time'] = pd.to_datetime(df['game_end_time'], format='%H:%M:%S', errors='coerce').dt.time



# 1. 독립 변수와 종속 변수 정의
X = df.drop(columns=["accident_score", "accident_count", "injury_count", "death_count", "region", "weekday", "sports_type", "sports_type"])
y = df['accident_score']


# 2. 야구
game_day_baseball = df[df['is_야구']]
no_game_day_baseball = df[~df['is_야구']]

formula = """accident_score ~ game_count + temperature + precipitation + snow_depth + is_post_season + is_hometeam_win + is_holiday + audience + 
        is_월 + is_화 + is_수 + is_목 + is_금 + is_토 + weather_condition_맑음 + weather_condition_비 + weather_condition_약간흐림 + weather_condition_흐림"""

formula2 = """accident_score ~ is_post_season + is_hometeam_win + audience + is_holiday + temperature + precipitation + snow_depth +
        weather_condition_맑음 + weather_condition_비 + weather_condition_약간흐림 + weather_condition_흐림"""


## 2-1. 야구 없는 날
model = smf.glm(
    formula=formula2,
    data=no_game_day_baseball,
    family=sm.families.Poisson()
)
result1_0 = model.fit()


## 2-2. 야구 있는 날
model = smf.glm(
    formula=formula2,
    data=game_day_baseball,
    family=sm.families.Poisson()
)
result1_1 = model.fit()


# 3. 축구
game_day_soccer = df[df['is_축구']]
no_game_day_soccer = df[~df['is_축구']]

## 3-1. 축구 없는 날
model = smf.glm(
    formula=formula2,
    data=no_game_day_soccer,
    family=sm.families.Poisson()
)
result2_0 = model.fit()

## 3-2. 축구 있는 날
model = smf.glm(
    formula=formula2,
    data=game_day_soccer,
    family=sm.families.Poisson()
)
result2_1 = model.fit()


# 4. 농구
game_day_basketball = df[df['is_농구']]
no_game_day_basketball = df[~df['is_농구']]

## 4-1. 농구 없는 날
model = smf.glm(
    formula=formula2,
    data=no_game_day_basketball,
    family=sm.families.Poisson()
)
result3_0 = model.fit()

## 4-2. 농구 있는 날
model = smf.glm(
    formula=formula2,
    data=game_day_basketball,
    family=sm.families.Poisson()
)
result3_1 = model.fit()


# 4. 배구
game_day_volleyball = df[df['is_배구']]
no_game_day_volleyball = df[~df['is_배구']]

## 4-1. 배구 없는 날
model = smf.glm(
    formula=formula2,
    data=no_game_day_volleyball,
    family=sm.families.Poisson()
)
result4_0 = model.fit()

## 4-2. 배구 있는 날
model = smf.glm(
    formula=formula2,
    data=game_day_volleyball,
    family=sm.families.Poisson()
)
result4_1 = model.fit()


# 5. 
## 5-1) 예측값 vs 실제값
df_model = df[['accident_score', 'audience', 'game_count', 'is_post_season', 'is_hometeam_win', 'game_start_time', 'game_end_time', 
               'is_holiday', 'temperature', 'precipitation', 'snow_depth', 'is_농구', 'is_배구', 'is_야구', 'is_여자배구', 'is_축구', 
               'is_금', 'is_목', 'is_수', 'is_월', 'is_일', 'is_토', 'is_화', 'weather_condition_맑음', 'weather_condition_비',
               'weather_condition_약간흐림', 'weather_condition_흐림']].copy()

df_model['predicted'] = result4_0.predict(df_model)
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


## 5-2) 잔차 vs 예측값
fig2= plt.figure(figsize=(8, 4))
sns.scatterplot(x='predicted', y='residuals', data=df_model)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted accident_score')
plt.ylabel('Residuals')
plt.title('잔차 vs 예측값')
plt.grid(True)
plt2 = fig2


## 5-3) 계수(Coefficient) 시각화
coef = result4_0.params
conf = result4_0.conf_int()
conf.columns = ['2.5%', '97.5%']
coef_df = pd.concat([coef, conf], axis=1).reset_index()
coef_df.columns = ['variable', 'coefficient', 'ci_lower', 'ci_upper']

fig3 = plt.figure(figsize=(8, 4))
sns.pointplot(data=coef_df, y='variable', x='coefficient', linestyles='')
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
st.write("* **목적**: 스포츠 경기 유무 및 종목에 따라 교통사고 위험도(`accident_score`)가 다른지, 그리고 어떤 요인들이 사고 위험 점수에 유의미한 영향을 주는지 정량적으로 평가합니다.")
st.write("* **모델**: 포아송 회귀 (Generalized Linear Model, `family=Poisson`, `link=log`)")
st.write("* **종속변수**: `accident_score` (사고 수와 심각도를 반영한 지표로 추정)")
st.write("""* **독립변수**:
    * 스포츠 관련 변수: `is_post_season` (포스트시즌 여부), `is_hometeam_win` (홈팀 승리 여부), `audience` (관중 수)
    * 시간 및 환경 변수: `weather_condition` (맑음, 비, 약간흐림, 흐림 - 기준 범주 대비), `is_holiday` (공휴일 여부), `temperature` (기온), `precipitation` (강수량), `snow_depth` (적설량)
    * 기타 영향 요인: `Intercept` (기본 사고 위험도)""")
st.write("")
st.write("")


st.header("2. 모델 적합도")
st.write("### 2-1. 스포츠 종목 별 경기 유무 Summary")
tabs = st.tabs(["야구⚾", "축구⚽", "농구🏀", "배구🏐"])

with tabs[0]:  
    col1, col2 = st.columns([1,1])
    with col1:
        st.write("#### 야구 경기가 없는 날")
        st.code(result1_0.summary(), language='text')
    with col2:
        st.write("#### 야구 경기가 있는 날")
        st.code(result1_1.summary(), language='text')
    

with tabs[1]:
    col1, col2 = st.columns([1,1])
    with col1:
        st.write("#### 축구 경기가 없는 날")
        st.code(result2_0.summary(), language='text')
    with col2:
        st.write("#### 축구 경기가 있는 날")
        st.code(result2_1.summary(), language='text')


with tabs[2]:
    col1, col2 = st.columns([1,1])
    with col1:
        st.write("#### 농구 경기가 없는 날")
        st.code(result3_0.summary(), language='text')
        st.write("농구 관중수 데이터셋을 수집하지 못해서 다른 스포츠에 비해 값이 좋지 않음")
    with col2:
        st.write("#### 농구 경기가 있는 날")
        st.code(result3_1.summary(), language='text')
        

with tabs[3]:
    col1, col2 = st.columns([1,1])
    with col1:
        st.write("#### 배구 경기가 없는 날")
        st.code(result4_0.summary(), language='text')
    with col2:
        st.write("#### 배구 경기가 있는 날")
        st.code(result4_1.summary(), language='text')

st.write("")
st.write("")

st.write("### 2-2. 전체 비교 Summary")
df1 = pd.DataFrame({
    "항목": [
        "No. Observations",
        "Log-Likelihood",
        "Deviance",
        "Pearson chi2",
        "Pseudo R-squ. (CS)"
    ],
    "야구 (경기 O)": [1810, -24356, 41560, 3.98e+04, 0.4821],
    "야구 (경기 X)": [26699, -3.2156e+05, 5.3605e+05, 5.54e+05, 0.1441],
    "축구 (경기 O)": [461, -5218.0, 8241.0, 9.03e+03, 0.2853],
    "축구 (경기 X)": [28048, -3.4058e+05, 5.6914e+05, 5.84e+05, 0.1700],
    "배구 (경기 O)": [550, -7269.6, 11951, 1.23e+04, 0.5547],
    "배구 (경기 X)": [27959, -3.3790e+05, 5.6417e+05, 5.80e+05, 0.1719],
    "해석": [
        "분석에 사용된 관측치 수",
        "높을수록(0에 가까울수록) 모델 적합도 양호",
        "낮을수록 모델 적합도 양호. 자유도 대비 값이 커서 과대산포 가능성이 있음.",
        "낮을수록 모델 적합도 양호. 값이 1보다 많이 커서 과대산포 가능성 있음.",
        "0과 1 사이 값, 1에 가까울수록 설명력 높음. 경기 있는 날 모델이 상대적으로 설명력이 더 높음."
    ]
})
styled_df = df1.style.set_table_styles([{'selector': 'th', 'props': [('background-color', '#dbeafe'), ('color', 'black')]}])
st.dataframe(styled_df, use_container_width=True)

st.write("""➡ **결론**:\n
* 모든 모델은 통계적으로 유의미한 결과를 보여줍니다 (Log-Likelihood 값). 전반적으로 **경기가 있는 날의 모델들이 경기가 없는 날의 모델들보다 Pseudo R-squared (CS) 값이 더 높게 나타나, 독립변수들이 사고 위험도를 더 잘 설명**하고 있음을 시사합니다. 예를 들어, 배구 경기 있는 날 모델(0.5547)의 설명력이 가장 높고, 야구 경기 있는 날 모델(0.4821)도 비교적 높은 설명력을 보입니다. 반면, 경기가 없는 날 모델들은 Pseudo R-squared 값이 0.14~0.17 수준으로 상대적으로 낮습니다.
* Deviance 및 Pearson chi2 값이 자유도(Df Residuals)에 비해 상당히 큰 것으로 보아, 모든 모델에서 과대산포(overdispersion)의 가능성이 존재합니다. 이는 실제 데이터의 분산이 포아송 분포의 가정(평균=분산)보다 크다는 것을 의미하며, 계수의 표준오차가 과소추정되어 p-value가 작게 나올 수 있습니다. 향후 분석 시 음이항 회귀 등을 고려할 수 있습니다.""")
st.write("")
st.write("")


st.write("### 2-3. 예측값과 실제값 비교")
st.pyplot(plt1)


st.write("### 2-3. 잔차 vs 예측값")
st.pyplot(plt2)


st.write("### 2-4. 계수(Coefficient) 시각화")
st.pyplot(plt3)

st.write("")
st.write("")

st.header("3. 주요 계수 해석")
st.write("""* `exp(β)`: 해당 변수가 1단위 증가할 때, `accident_score`의 평균이 `exp(β)` 배만큼 변하는 것을 의미.
    * `exp(β) > 1`: 위험도 증가
    * `exp(β) < 1`: 위험도 감소
    * 예: `β = 0.05` 이면 `exp(0.05) ≈ 1.051`, 즉 5.1% 증가. `β = -0.1` 이면 `exp(-0.1) ≈ 0.905`, 즉 9.5% 감소.""")

df2 = pd.DataFrame({
    "변수명": [
        "Intercept (절편)",
        "audience",
        "is_holiday",
        "temperature",
        "precipitation/snow_depth"
    ],
    "대표적 경향 (β 부호)": [
        "경기 O > 경기 X",
        "경기 O: 대부분 +",
        "일관되게 -",
        "혼재",
        "대부분 -"
    ],
    "P-value": [
        "<0.001",
        "<0.001",
        "<0.001",
        "다양함",
        "다양함"
    ],
    "해석 요약 (일반적 경향 및 특이점)": [
        "경기가 있는 날의 기본 사고 위험도(exp(β))가 경기가 없는 날보다 모든 스포츠에서 약 1.5배~1.6배 높음. (예: 야구 exp(3.52) ≈ 33.8 vs exp(3.10) ≈ 22.2)",
        "경기 있는 날: 관중 수 증가는 야구, 배구에서 사고 위험도 증가. 축구는 특이하게 관중 증가 시 위험도 감소(-2.6e-06). 경기 없는 날: 전반적으로 양의 관계(일반적 교통량 반영 가능성).",
        "모든 모델에서 공휴일은 사고 위험도를 낮춤 (약 14~16% 감소 효과, exp(-0.15) ≈ 0.86).",
        "경기 있는 날: 야구/축구는 기온 상승 시 위험도 감소, 배구는 증가. 경기 없는 날: 기온 상승 시 위험도 소폭 증가.",
        "강수량/적설량 증가는 대부분 모델에서 사고 위험도를 낮춤. 이는 악천후 시 운전자가 더 조심하거나 통행량 자체가 줄어들기 때문일 수 있음."
    ]
})
styled_df = df2.style.set_table_styles([{'selector': 'th', 'props': [('background-color', '#dbeafe'), ('color', 'black')]}])
st.dataframe(styled_df, use_container_width=True)
st.write("""➡ **핵심 해석**:
* **경기 유무가 가장 큰 영향**: 모든 스포츠에서 경기가 있는 날의 기본 `accident_score` (Intercept)가 경기가 없는 날보다 약 1.5~1.6배 높습니다. 이는 스포츠 경기 자체가 주변 교통사고 위험을 높이는 주요 요인임을 시사합니다.
* **관중 효과**: 야구와 배구에서는 관중 수가 많을수록 사고 위험이 증가하는 반면, 축구에서는 오히려 관중 수가 많을수록 사고 위험이 소폭 감소하는 특이한 결과가 나타났습니다.
* **날씨의 복합적 영향**: 경기 있는 날, 특히 야구와 배구에서는 궂은 날씨(비, 흐림 등)가 오히려 사고 위험도를 낮추는 경향이 있는데, 이는 관중 감소 또는 운전자들의 주의력 증가 때문일 수 있습니다. 반면 경기 없는 날에는 일반적인 예상과 유사하게 일부 궂은 날씨가 위험도를 높입니다.
* **공휴일 효과**: 공휴일은 모든 경우에 사고 위험도를 일관되게 낮추는 효과가 있었습니다.
""")
st.write("")
st.write("")


st.header("4. 모델 특이점 및 비교")
st.write("""1. **경기 유무에 따른 `Intercept` 차이**:
    * 야구: 경기 O (3.52) vs 경기 X (3.11) → `exp(3.52)/exp(3.11)` ≈ 1.51배
    * 축구: 경기 O (3.56) vs 경기 X (3.11) → `exp(3.56)/exp(3.11)` ≈ 1.57배
    * 배구: 경기 O (3.53) vs 경기 X (3.08) → `exp(3.53)/exp(3.08)` ≈ 1.57배
    * 모든 스포츠에서 경기가 있는 날의 기본 사고 위험이 경기가 없는 날보다 약 50~60% 높게 시작합니다.""")
st.write("""2. **스포츠 종목별 `audience` 효과 (경기 있는 날)**:
    * 야구: `coef = 1.046e-05` (양수). 관중 1만 명 증가 시 사고위험 `exp(1.046e-05 * 10000)` = `exp(0.1046)` ≈ 1.11배 (약 11% 증가).
    * 축구: `coef = -2.604e-06` (음수). 관중 1만 명 증가 시 사고위험 `exp(-2.604e-06 * 10000)` = `exp(-0.02604)` ≈ 0.974배 (약 2.6% 감소).
    * 배구: `coef = 3.26e-05` (양수). 관중 1만 명 증가 시 사고위험 `exp(3.26e-05 * 10000)` = `exp(0.326)` ≈ 1.385배 (약 38.5% 증가).
    * 배구의 관중 당 위험도 증가율이 가장 크며, 축구는 유일하게 음의 관계를 보입니다.""")
st.write("""3. **`is_post_season` 효과 (경기 있는 날)**:
    * 야구: `coef = 0.4059`. 포스트시즌 시 사고 위험 약 `exp(0.4059)` ≈ 1.50배 (약 50% 증가).
    * 축구: `coef = -0.0702`. 포스트시즌 시 사고 위험 약 `exp(-0.0702)` ≈ 0.932배 (약 6.8% 감소).
    * 배구: P-value > 0.05로 유의미하지 않음.
    * 야구 포스트시즌의 위험도 증가가 두드러집니다.""")
st.write("""4. **날씨 변수의 상반된 효과**:
    * **경기 있는 날 (야구, 배구)**: `weather_condition_맑음[T.True]`, `weather_condition_비[T.True]` 등이 모두 음수 계수를 가집니다. 이는 기준 날씨(예: '구름조금' 등 명시되지 않은 기본값)보다 맑거나 비가 오거나 흐린 날에 사고 위험이 낮아짐을 의미합니다. 이는 경기 당일 날씨가 좋지 않으면 관람객 수가 줄거나, 운전자들이 더 주의하기 때문일 수 있습니다.
    * **경기 없는 날**: 대체로 `weather_condition_맑음[T.True]`, `weather_condition_비[T.True]` 등이 양수 계수를 가져, 기준 날씨보다 사고 위험이 높아짐을 의미합니다. 이는 일반적인 교통 상황에서의 날씨 효과와 유사합니다.""")
st.write("""5. **과대산포 (Overdispersion)**: 모든 모델에서 `Scale`이 1.0000으로 고정되어 있지만, `Pearson chi2 / Df Residuals` 비율이 1보다 훨씬 큽니다 (예: 야구 경기 O의 경우 39800 / 1799 ≈ 22.1). 이는 데이터의 분산이 평균보다 크다는 신호로, 포아송 모델의 가정을 위배할 수 있습니다. 이 경우 표준오차가 과소평가되어 변수들이 실제보다 더 유의하게 나올 수 있습니다.""")
st.write("")
st.write("")


st.header("5. 결론 및 제언")
st.write("##### 🎯 핵심 결론")
st.write("""1.  **스포츠 경기 개최는 그 자체로 주변 지역의 교통사고 위험도(`accident_score`)를 평균적으로 약 1.5~1.6배 높이는 가장 강력한 요인입니다.**
2.  **관중 수**는 야구와 배구 경기가 있는 날 사고 위험을 증가시키는 반면, 축구 경기가 있는 날에는 오히려 소폭 감소시키는 독특한 양상을 보였습니다. 이는 축구 경기 시의 교통 관리나 안전 대책이 상대적으로 더 효과적일 수 있음을 시사합니다.
3.  **포스트시즌**은 야구 경기 시 위험도를 크게 높이지만(약 50% 증가), 축구 경기 시에는 오히려 낮추는 효과가 있었습니다. 배구는 유의미한 영향이 없었습니다.
4.  **날씨**의 영향은 경기 유무에 따라 다르게 나타납니다. 경기 당일에는 (특히 야구/배구) 궂은 날씨가 오히려 사고 위험을 낮추는 경향이 관찰되는데, 이는 관중 수 감소 또는 운전 행태 변화와 관련될 수 있습니다.
5.  **공휴일**은 일관되게 사고 위험을 낮추는 요인으로 작용했습니다.
6.  **강수량/적설량** 증가는 대부분 사고 위험을 낮추는 것으로 나타나, 악천후 시 교통량 감소 또는 운전자들의 경각심 증가를 반영할 수 있습니다.""")

st.write("##### ✅ 향후 제언")
st.write("""1.  **맞춤형 교통 관리 전략 수립**:
    *   모든 스포츠 경기 당일, 특히 야구 포스트시즌 경기 시에는 강화된 교통 관리 및 안전 대책이 필요합니다.
    *   배구 경기 시에는 관중 수 증가에 따른 위험도 상승폭이 크므로, 관중 규모에 따른 탄력적 대응이 중요합니다.
    *   축구 경기 시 관중 증가가 위험도 감소로 이어진 원인(예: 특정 교통 통제 방식, 대중교통 이용 유도 캠페인 등)을 파악하여 다른 스포츠에도 적용할 수 있는지 검토가 필요합니다.""")
st.write("""2.  **모델 개선**:
    *   모든 모델에서 관찰된 **과대산포(overdispersion)** 문제를 해결하기 위해, 포아송 회귀 대신 **음이항 회귀(Negative Binomial Regression)** 모델을 적용하여 결과의 강건성(robustness)을 높일 필요가 있습니다.
    *   날씨 변수의 경우, 단순 기상 상태 외에 '경기 당일 예보된 날씨'와 '실제 날씨'의 차이, 또는 특정 기상 조건과 관중 수 간의 **상호작용 항(interaction term)**을 모델에 추가하여 보다 정교한 분석을 시도할 수 있습니다.
    *   `audience` 변수는 경기 없는 날에는 0이거나 매우 낮을 것으로 예상되는데, 해당 모델에서 유의하게 나온 것은 지역 내 다른 유동인구를 반영할 수 있으므로 데이터 수집 범위를 명확히 할 필요가 있습니다.""")
st.write("""3.  **심층 분석**:
    *   축구 경기에서 관중 수가 많을수록 사고 위험이 낮아지는 현상, 강수/적설 시 사고 위험이 낮아지는 현상에 대해 추가적인 정성적, 정량적 연구(예: 설문조사, 현장 관찰, 세부 교통 데이터 분석)가 필요합니다.
    *   `is_hometeam_win` 변수의 경우, 경기 후 시간대별 사고 발생 패턴과 연관지어 분석하면 더 의미 있는 해석이 가능할 수 있습니다 (예: 승리 후 축하 행위로 인한 위험 증가 등).""")
