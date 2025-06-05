import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date, timedelta, time 
import json

st.image("./06.web_ui/images/writing-7702615_2.jpg", use_container_width=True)

st.title("날짜별 교통사고 통계")

tabs = st.tabs(["연도별(Year)", "월별(Month)", "요일별(Day)"])

with tabs[0]:
    st.header("연도별 통계")
    st.write("연도별 교통사고 통계 내용을 여기에 작성")

with tabs[1]:
    st.header("월별 통계")
    st.write("월별 교통사고 통계 내용을 여기에 작성")

with tabs[2]:
    st.header("요일별 통계")
    st.write("요일별 교통사고 통계 내용을 여기에 작성")