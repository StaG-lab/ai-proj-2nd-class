import streamlit as st

def set_config():
    st.set_page_config(
        initial_sidebar_state="expanded",
        page_icon="🚗",
        layout="wide",
        page_title="스포츠 교통사고 예방"
    )

    st.sidebar.page_link("Home.py", label="🏡 메인페이지")
    st.sidebar.page_link("pages/경기 유무에 따른 교통사고율 비교.py", label="1️⃣ 경기 유무에 따른 교통사고율 비교")
    st.sidebar.page_link("pages/스포츠 종목별 교통사고율 비교.py", label="2️⃣ 스포츠 종목별 교통사고율 비교")
    st.sidebar.page_link("pages/경기 중요도에 따른 교통사고율 비교.py", label="3️⃣ 경기 중요도에 따른 교통사고율 비교")
    st.sidebar.page_link("pages/🧙 지역별 날짜별 교통사고 분석.py", label="🧙🏻‍♂️ 데이터셋 마법사")
    st.sidebar.write("---")
    st.sidebar.write("기타 페이지 (임시)")
    st.sidebar.page_link("pages/지역별 교통사고 통계 (검색).py", label="🚞 지역별 교통사고 통계")

    return


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
                        st.toast("로그아웃 되었습니다.")
                        st.rerun()
            else:
                if st.button("로그인"):
                    st.page_link("pages/Signin.py", label="로그인하러 가기")
  
