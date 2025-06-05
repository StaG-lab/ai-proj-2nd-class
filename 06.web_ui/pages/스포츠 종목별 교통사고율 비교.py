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