import streamlit as st
from utils.layout import set_config
from utils.auth import signup_user
import re


# 전화번호 유효성 검사 함수
def is_valid_phone_number(phone_number: str) -> bool:
    pattern = r'^010-\d{4}-\d{4}$'
    return re.match(pattern, phone_number) is not None


set_config()
st.title("📝 회원가입")

userid = st.text_input("아이디")
userpw = st.text_input("비밀번호", type="password")
userpw_confirm = st.text_input("비밀번호 확인", type="password")
name = st.text_input("이름")
phone_number = st.text_input("전화번호 (예: 010-1234-5678)")

if st.button("회원가입"):
    if not userid or not userpw or not name or not phone_number:
        st.error("모든 항목을 입력해주세요.")
    elif userpw != userpw_confirm:
        st.error("비밀번호가 일치하지 않습니다.")
    elif not is_valid_phone_number(phone_number):
        st.error("전화번호 형식이 올바르지 않습니다. 예: 010-1234-5678")
    else:
        success, msg = signup_user(userid, userpw, name, phone_number)
        if success:
            st.success(msg)
        else:
            st.error(msg)
