# /05.modeling/1차_데이터셋_선형회기_모델링_고승규.ipynb

import streamlit as st
from utils.layout import set_config, login_widget
from PIL import Image

set_config()
st.title("경기 유무에 따른 교통사고율 비교")
login_widget()

st.write("> ##### 스포츠 경기가 있는 날에는 교통사고가 더 많이 일어날까?")

tabs = st.tabs(["☝🏻 통계적 관점에서의 분석", 
                "✌🏻 선형회귀모델을 통한 분석", 
                "👌🏻 포아송회귀모델을 통한 분석"])

with tabs[0]:
    st.write(1)

with tabs[1]:
    st.write("* 입력 변수(X) : `match_정규시즌`, `match_포스트시즌`, `is_holiday`, `start_hour`, `audience`")
    st.write("* 종속 변수(y) : `accident_count`(교통사고 건수)")
    st.write("")
    st.write("")

    st.header("스포츠 경기에 따른 교통사고 회귀계수")
    plot = Image.open('images/야구_선형회귀_plot.png')
    st.image(plot)
    st.write("* 평균제곱오차(MSE) : 18.251393756378654")
    st.write("* 결정계수(R²) : 0.04765667254865613")
    st.write("* 양의 상관관계 : `audience`, `start_hour`, `match_포스트시즌`")
    st.write("* 음의 상관관계 : `is_holiday`")
    st.write("")
    st.write("**😎 관중수, 포스트시즌 매치, 경기 시작 시간이 유의미한 영향을 미침**")
    st.write("**😣 하지만 결정계수가 너무 낮아 교통사고건수 변화를 충분히 설명하지 못함**")
    st.write("")
    st.write("")

    st.header("Summary")
    summary = """                 Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:         accident_count   No. Observations:                 2072
Model:                            GLM   Df Residuals:                     2066
Model Family:                 Poisson   Df Model:                            5
Link Function:                    Log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -7003.7
Date:                Wed, 04 Jun 2025   Deviance:                       8393.5
Time:                        17:17:25   Pearson chi2:                 7.95e+03
No. Iterations:                     5   Pseudo R-squ. (CS):             0.1487
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          z      P>|z|      [0.025      0.975]
-------------------------------------------------------------------------------
Intercept       0.6007      0.150      4.012      0.000       0.307       0.894
match_정규시즌      0.0340      0.062      0.549      0.583      -0.087       0.156
match_포스트시즌     0.4069      0.084      4.827      0.000       0.242       0.572
is_holiday     -0.2280      0.029     -7.933      0.000      -0.284      -0.172
start_hour      0.0377      0.010      3.749      0.000       0.018       0.057
audience     2.629e-05   1.94e-06     13.570      0.000    2.25e-05    3.01e-05
==============================================================================="""
    st.code(summary, language='text')
    st.write("""
             * 절편(Intercept) : 
                * $$exp(0.6007)≈1.823$$
                * 모든 변수의 값이 0일 때 예상 교통사고 건수 1.82건
             """)
    st.write("""
             * `audience`
                * 변수의 계수가 가장 크고, 통계적으로 유의함
                * $$exp(0.00002629×1000)≈exp(0.02629)≈1.0266$$
                * **관중이 1000명 증가하면 교통사고는 약 2.66% 증가할 것으로 예상**
             """)
    st.write("""
             * `match_포스트시즌`
                * $$exp(0.4069)≈1.502$$
                * **포스트시즌 경기라면 사고건수가 약 50.2% 증가할 것으로 예상**
             """)
    st.write("""
             * `is_holiday`
                * $$exp(−0.2280)≈0.796$$
                * **공휴일에는 사고건수가 약 20.4% 감소할 것으로 예상**
             """)


with tabs[2]:
    st.write(3)
