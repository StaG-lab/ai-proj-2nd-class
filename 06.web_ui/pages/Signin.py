import streamlit as st
from utils.layout import set_config
from utils.auth import authenticate_user

set_config()

st.title("🔐 로그인")

# 로그인 폼
with st.form("login_form"):
    userid = st.text_input("아이디", key="userid_input")
    userpw = st.text_input("비밀번호", type="password", key="userpw_input")
    submitted = st.form_submit_button("로그인")

if submitted:
    user = authenticate_user(userid, userpw)
    if user:
        st.success(f"{user.name}님 환영합니다!")
        st.session_state["name"] = user.name
        st.page_link("Home.py", label="메인 페이지로 이동하기", icon="▶")
    else:
        st.error("아이디 또는 비밀번호가 잘못되었습니다.")
