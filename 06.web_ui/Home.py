import streamlit as st
from utils.layout import set_config, login_widget


set_config()
st.image("./06.web_ui/images/naj-eneun-bin-gyeong-gijang2.jpg", use_container_width=True)
st.markdown(
    """
    <h1 style="text-align:center;color:white;background-color:#6B8E23;padding:10px;border-radius:5px;">
    ⚽ 스포츠 경기 교통사고 예방 ⚾
    </h1>
    """,
    unsafe_allow_html=True
)
st.write("")

# 화면 구성
col1, col2 = st.columns([2,1])
with col1:
    st.write("")
    st.header("이곳은 메인 페이지입니다.")
with col2:
    # 로그인 폼
    with st.form("login_form"):
        userid = st.text_input("**아이디**", key="userid_input")
        userpw = st.text_input("**비밀번호**", type="password", key="userpw_input")
        submitted = st.form_submit_button("로그인")

    if submitted:
        user = authenticate_user(userid, userpw)
        if user:
            st.success(f"{user.name}님 환영합니다!")
            st.session_state["name"] = user.name
            st.page_link("Home.py", label="메인 페이지로 이동하기", icon="▶")
        else:
            st.error("아이디 또는 비밀번호가 잘못되었습니다.")


