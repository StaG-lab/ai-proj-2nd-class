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

def login_widget():
    st.markdown("""
    <div style="background-color:#f5f5f5; padding:20px; border-radius:5px;">
        <h3>🔒 로그인</h3>
        <input type="text" placeholder="아이디">
        <input type="password" placeholder="비밀번호">
        <button>로그인</button>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([2,1])
with col1:
    st.write("")
    st.header("이곳은 메인 페이지입니다.")
with col2:
    login_widget()
