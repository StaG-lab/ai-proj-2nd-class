import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pymysql
import holiday
import time
from sqlalchemy import create_engine, text

st.title("스포츠 종목별 교통사고 통계")

tabs = st.tabs(["야구⚾", "농구🏀", "축구⚽", "배구🏐", "여자배구🏐"])

with tabs[0]:
    st.header("야구 통계")
    st.write("야구 관련 교통사고 통계 내용을 여기에 작성")

with tabs[1]:
    st.header("농구 통계")
    st.write("농구 관련 교통사고 통계 내용을 여기에 작성")

with tabs[2]:
    st.header("축구 통계")
    st.write("축구 관련 교통사고 통계 내용을 여기에 작성")

with tabs[3]:
    st.header("배구 통계")
    st.write("배구 관련 교통사고 통계 내용을 여기에 작성")

with tabs[4]:
    st.header("여자자배구 통계")
    st.write("여자자배구 관련 교통사고 통계 내용을 여기에 작성")