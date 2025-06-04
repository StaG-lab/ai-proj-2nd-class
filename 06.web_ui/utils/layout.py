import streamlit as st

def set_config():
    st.set_page_config(
        initial_sidebar_state="collapsed",
        page_icon="🚗",
        layout="wide",
        page_title="스포츠 교통사고 예방"
    )

def login_widget():
    placeholder = st.empty()

    with placeholder.container():
        cols = st.columns([3, 1])        
        with cols[1]:
            if st.session_state.get("name"):
                cols2 = st.columns([1, 1])
                with cols2[0]:
                    name = st.session_state.get("name", "")
                    st.markdown(f"환영합니다, {name}님👋")
                with cols2[1]:
                    if st.button("로그아웃"):
                        st.session_state.clear()
                        st.rerun()
            else:
                if st.button("로그인"):
                    st.page_link("pages/Signin.py", label="로그인하러 가기")


def sidebar_widget():
    return
