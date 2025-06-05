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

st.image("./06.web_ui/images/accident-5167244_2.jpg", use_container_width=True)

st.title("지역별 교통사고 통계")

st.write("")

tabs = st.tabs(["검색창","수원(예시)", "송파(예시)"])

with tabs[0]:
    # DB 연결
    with open("./06.web_ui/db_config.json", "r") as f:
        config = json.load(f)

    DB_USER = config["DB_USER"]
    DB_PASSWORD = config["DB_PASSWORD"]
    DB_HOST = config["DB_HOST"]
    DB_PORT = config["DB_PORT"]
    DB_NAME = config["DB_NAME"]

    engine_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # 1️⃣ "데이터 연결중..." 메시지를 표시
    loading_message = st.empty()
    loading_message.info("**데이터 연결중...**")

    try:
        time.sleep(2)  # 시뮬레이션 (연결 시간)
        loading_message.empty()
        engine = create_engine(engine_url)
        st.success("**MySQL 데이터베이스에 성공적으로 연결되었습니다.**")
    except Exception as e:
        st.error(f"**데이터베이스 연결 오류: {e}**")
        st.stop()

        # 추가 분석 코드 작성


    # --- 데이터베이스에서 데이터 로드하는 함수 ---
    def load_table_to_df(table_name, engine):
        """**지정된 테이블에서 모든 데이터를 Pandas DataFrame으로 로드합니다.**"""
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(text(query), engine.connect())
            st.write(f"**'{table_name}' 테이블 로드 완료. {len(df)} 행.**")
            # 날짜/시간 컬럼 타입 변환 (필요시)
            if 'game_date' in df.columns:
                df['game_date'] = pd.to_datetime(df['game_date'])
            if 'accident_date' in df.columns:
                df['accident_date'] = pd.to_datetime(df['accident_date'])
            if 'weather_date' in df.columns:
                df['weather_date'] = pd.to_datetime(df['weather_date'])
            # start_time, end_time, weather_time은 문자열로 로드될 수 있으므로, 필요시 time 객체로 변환
            return df
        except Exception as e:
            st.info(f"**'{table_name}' 테이블 로드 중 오류 발생: {e}**")
            return pd.DataFrame() # 오류 발생 시 빈 DataFrame 반환

    # --- 데이터 로드 ---
    stadium_df = load_table_to_df('stadium', engine)
    sports_game_df = load_table_to_df('sports_game', engine)
    traffic_accident_df = load_table_to_df('traffic_accident', engine)
    weather_df = load_table_to_df('weather', engine)

    # 데이터베이스 연결 종료
    if 'engine' in locals() and engine:
        engine.dispose()
        st.write("\n**데이터베이스 연결이 종료되었습니다.**")

    # 사용자 입력
    START_DATE = pd.to_datetime("2023-01-01")  # 분석 시작일
    END_DATE = pd.to_datetime("2024-12-31")    # 분석 종료일

    stadium_region = stadium_df['region'].unique().tolist()

    # 사용자 입력 (텍스트)
    input_region = st.text_input("**데이터셋을 구성할 지역명을 입력하세요:**")

    TARGET_REGION = None
    if input_region:
        for r in stadium_region:
            r_split = r.split(' ')[1]  # e.g., "서울특별시 강남구" → "강남구"
            if input_region in r_split:
                TARGET_REGION = r
                break
        if TARGET_REGION:
            st.success(f"선택한 지역: {TARGET_REGION}")
            date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
            base_df = pd.DataFrame({'date': date_range})
            base_df['region_code'] = TARGET_REGION
            st.write(base_df.head())
        else:
            st.error(f"**입력하신 '{input_region}'은 정보가 없습니다. 다시 입력해주세요.**")


    # TARGET_REGION에 따른 데이터 셋 구성

    # 스타디움 정보 가져오기
    if not stadium_df.empty:
        stadiums_in_target_region = stadium_df[stadium_df['region'] == TARGET_REGION]
        stadium_codes_in_region = stadiums_in_target_region['stadium_code'].unique().tolist()
        st.write(f"\n**{TARGET_REGION} 내 경기장 코드: {stadium_codes_in_region}**")
    else:
        stadium_codes_in_region = []
        st.write(f"\n**{TARGET_REGION} 내 경기장 정보 없음 또는 stadium 테이블 로드 실패.**")

    # 스포츠경기 정보 가져오기
    if not sports_game_df.empty and stadium_codes_in_region:
        games_in_region_df = sports_game_df[sports_game_df['stadium_code'].isin(stadium_codes_in_region)]
        games_in_region_df = games_in_region_df.rename(columns={'game_date': 'date'})
        #st.write(games_in_region_df)
        if not games_in_region_df.empty:
            game_summary_df = games_in_region_df.groupby('date').agg(
                game_count=('stadium_code', 'count'),
                sports_type_list=('sports_type', lambda x: list(set(x))),
                has_playoff_list=('match_type', lambda x: 1 if any('플레이오프' in str(mt).lower() for mt in x) else 0)
            ).reset_index()
            
            game_summary_df['sports_type'] = game_summary_df['sports_type_list'].apply(lambda x: ','.join(sorted(x)) if x else '없음')
            game_summary_df['has_playoff'] = game_summary_df['has_playoff_list']
            game_summary_df = game_summary_df[['date', 'game_count', 'sports_type', 'has_playoff']]
            base_df = pd.merge(base_df, game_summary_df, on='date', how='left')
        else:
            st.write(f"{TARGET_REGION} 내 해당 기간 경기 정보 없음.")
            base_df['game_count'] = 0
            base_df['sports_type'] = '없음'
            base_df['has_playoff'] = 0
    else:
        st.write("sports_game_df 로드 실패 또는 대상 지역 내 경기장 없음.")
        base_df['game_count'] = 0
        base_df['sports_type'] = '없음'
        base_df['has_playoff'] = 0

    base_df['game_count'] = base_df['game_count'].fillna(0).astype(int)
    base_df['sports_type'] = base_df['sports_type'].fillna('없음')
    base_df['has_playoff'] = base_df['has_playoff'].fillna(0).astype(int)

    # 5. 교통 사고 데이터 가져오기
    if not traffic_accident_df.empty:
        accidents_in_region_df = traffic_accident_df[traffic_accident_df['region'] == TARGET_REGION]
        accidents_in_region_df = accidents_in_region_df.rename(columns={'accident_date': 'date'})
        if not accidents_in_region_df.empty:
            accident_summary_df = accidents_in_region_df.groupby('date').agg(
                accident_count_sum=('accident_count', 'sum')
            ).reset_index()
            accident_summary_df = accident_summary_df.rename(columns={'accident_count_sum': 'accident_count'})
            
            base_df = pd.merge(base_df, accident_summary_df, on='date', how='left')
        else:
            st.write(f"**{TARGET_REGION} 내 해당 기간 교통사고 정보 없음.**")
            base_df['accident_count'] = 0
    else:
        st.write("**traffic_accident_df 로드 실패.**")
        base_df['accident_count'] = 0
        
    base_df['accident_count'] = base_df['accident_count'].fillna(0).astype(int)

    # 날씨 데이터 가져오기
    if not weather_df.empty:
        weather_region_df = weather_df[weather_df['region'] == "수원"]
        weather_region_df = weather_region_df.rename(columns={'weather_date': 'date'})

        if not weather_region_df.empty:
            # 날씨 데이터는 하루에 여러 번 기록될 수 있으므로, 일별 집계 필요
            weather_summary_df = weather_region_df.groupby('date').agg(
                temperature=('temperature', 'mean'),
                precipitation=('precipitation', 'sum'),
                snow_depth=('snow_depth', 'sum'),
                avg_cloud_amount=('cloud_amount', 'mean') # 대표 날씨 상태 추론용
            ).reset_index()

            def get_weather_condition(row):
                if (pd.isna(row['precipitation']) or pd.isna(row['snow_depth'])) and pd.isna(row['avg_cloud_amount']):
                    return '정보없음' # 데이터가 없는 경우
                if row['precipitation'] > 0 or pd.isna(row['snow_depth']) > 0:
                    return '비/눈'
                elif pd.notna(row['avg_cloud_amount']):
                    if row['avg_cloud_amount'] >= 7: # (0-10 기준)
                        return '흐림'
                    elif row['avg_cloud_amount'] >= 3:
                        return '구름조금' # 또는 '약간흐림' 등
                    else:
                        return '맑음'
                return '정보없음' # 강수량 없고 구름 정보도 없는 경우
                
            weather_summary_df['weather_condition'] = weather_summary_df.apply(get_weather_condition, axis=1)
            weather_summary_df = weather_summary_df[['date', 'temperature', 'precipitation', 'snow_depth', 'weather_condition']]
            
            base_df = pd.merge(base_df, weather_summary_df, on='date', how='left')
        else:
            st.write(f"{TARGET_REGION} 내 해당 기간 날씨 정보 없음.")
            base_df['temperature'] = pd.NA
            base_df['precipitation'] = pd.NA
            base_df['snow_depth'] = pd.NA
            base_df['weather_condition'] = '정보없음'
    else:
        st.write("weather_df 로드 실패.")
        base_df['temperature'] = pd.NA
        base_df['precipitation'] = pd.NA
        base_df['snow_depth'] = pd.NA
        base_df['weather_condition'] = '정보없음'

    # 공휴일 및 주말 데이터 가져오기
    weekday_map_kr = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
    base_df['weekday'] = base_df['date'].dt.dayofweek.map(weekday_map_kr)

    try:
        import holidays
        # base_df['date']에서 연도를 뽑아와서 unique 값으로 추출
        kr_holidays = holidays.KR(years=base_df['date'].dt.year.unique().tolist()) 
        is_statutory_holiday = base_df['date'].apply(lambda d: d in kr_holidays)
        is_saturday = (base_df['weekday'] == '토')
        is_sunday = (base_df['weekday'] == '일')
        # 3. 세 가지 조건을 OR 연산자로 결합하여 is_holiday 컬럼 생성
        # (하나라도 True이면 True -> 1, 모두 False이면 False -> 0)
        base_df['is_holiday'] = (is_statutory_holiday | is_saturday | is_sunday).astype(int)
    except ImportError:
        st.write("**`holidays` 라이브러리가 설치되지 않았습니다. `pip install holidays`로 설치해주세요. 'is_holiday'는 0으로 처리됩니다.**")
        base_df['is_holiday'] = 0
    except Exception as e:
        st.write(f"**공휴일 정보 처리 중 오류: {e}. 'is_holiday'는 0으로 처리됩니다.**")
        base_df['is_holiday'] = 0
        
    # 최종 데이터 셋 구성
    final_df = base_df[[
        'date', 'region_code', 'accident_count', 'game_count', 'sports_type',
        'has_playoff', 'temperature', 'precipitation', 'snow_depth', 'weather_condition',
        'is_holiday', 'weekday'
    ]].copy() # SettingWithCopyWarning 방지를 위해 .copy() 사용

    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    final_df['temperature'] = pd.to_numeric(final_df['temperature'], errors='coerce').round(1)
    final_df['precipitation'] = pd.to_numeric(final_df['precipitation'], errors='coerce').round(1)

    # Set Korean font for matplotlib
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False # To display minus sign correctly

    df = final_df.copy()

    st.markdown("## 1. 초기 데이터 검사 <span style='font-size:16px; color:gray'>_(Initial Data Inspection)_</span>", unsafe_allow_html=True)

    st.write("Shape of the dataframe:", df.shape)
    st.write("\nFirst 5 rows:")
    st.write(df.head())
    st.write("\nData types and non-null counts:")
    df.info()
    st.write("\nSummary statistics for numerical columns:")
    st.write(df.describe())
    st.write("\nSummary statistics for object columns:")
    st.write(df.describe(include='object'))
    st.write("\nMissing values per column:")
    st.write(df.isnull().sum())

    st.markdown("## 2. 데이터 정제 및 전처리 <span style='font-size:16px; color:gray'>_(Data Cleaning and Preprocessing)_</span>", unsafe_allow_html=True)

    # Convert 'date' to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # Convert boolean-like columns to actual booleans for clarity if needed (0/1 is fine for most analyses)
    df['has_playoff'] = df['has_playoff'].astype(bool)
    df['is_holiday'] = df['is_holiday'].astype(bool)

    # Create a column 'has_game' for easier filtering
    df['has_game'] = df['game_count'] > 0

    # Check consistency: region_code
    st.write("\nUnique values in 'region_code':", df['region_code'].unique())
    # If 'region_code' is constant, we might drop it for modeling, but keep for EDA context.

    # Clean sports_type: "없음" means no sport, otherwise it's a list
    # For analysis, we can create dummy variables for common sports
    # Step 1. sports_type에서 고유한 스포츠 추출
    all_sports = set()
    for sports_list in df['sports_type'].unique():
        if sports_list != "없음":
            for sport in sports_list.split(','):
                all_sports.add(sport.strip())

    # Step 2. 정렬
    all_sports_sorted = sorted(list(all_sports))

    # Step 3. 마크다운 형태로 출력
    markdown_list = "\n".join([f"- {sport}" for sport in all_sports_sorted])
    st.markdown(f"""
    **Identified unique sport types:**

    {markdown_list}
    """)


    # Create boolean columns for each identified sport
    for sport in sorted(list(all_sports)):
        df[f'is_{sport}'] = df['sports_type'].apply(lambda x: sport in x)

    st.write("\nDataFrame with new sport boolean columns (first 5 rows):")
    st.write(df.head())

    st.markdown("## 3. 피처 엔지니어링 <span style='font-size:16px; color:gray'>_(Feature Engineering (Time-based features))_</span>", unsafe_allow_html=True)

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week_num'] = df['date'].dt.dayofweek # Monday=0, Sunday=6
    df['day_name'] = df['date'].dt.day_name()
    # Verify 'weekday' column consistency (optional, if 'weekday' is already reliable)
    # df['day_name_korean'] = df['date'].dt.strftime('%a') # Mon, Tue... in Korean locale (if set)

    # Map day_of_week_num to Korean day names for consistency with provided 'weekday'
    day_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
    df['derived_weekday'] = df['day_of_week_num'].map(day_map)

    # Check if derived_weekday matches the provided weekday column
    if 'weekday' in df.columns:
        mismatch_count = (df['derived_weekday'] != df['weekday']).sum()
        st.write(f"\nNumber of mismatches between derived weekday and provided weekday: {mismatch_count}")
        if mismatch_count > 0:
            st.write("Consider using the derived weekday or thoroughly checking the provided one.")
            # For now, we'll prioritize the derived one for consistency if there's a discrepancy
            # Or, trust the provided one if it's confirmed correct. Let's use the derived one.
            df['weekday_final'] = df['derived_weekday']
        else:
            df['weekday_final'] = df['weekday'] # or derived_weekday, they are the same
    else:
        df['weekday_final'] = df['derived_weekday']

    st.markdown("## 4. 단변량 분석 <span style='font-size:16px; color:gray'>_(Univariate Analysis)_</span>", unsafe_allow_html=True)

    # 4.1 Distribution of Accident Count
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['accident_count'], kde=True, bins=20)
    ax.set_title('Distribution of Daily Accident Counts')
    ax.set_xlabel('Number of Accidents')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df['accident_count'])
    ax.set_title('Box Plot of Daily Accident Counts')
    ax.set_xlabel('Number of Accidents')
    ax.grid(True)
    st.pyplot(fig)

    desc = df['accident_count'].describe().round(2).to_frame()
    desc.columns = ['Accident count statistics']  # 컬럼명 예쁘게 바꾸기
    st.table(desc)

    # 4.2 Distribution of Numerical Weather Features
    weather_cols = ['temperature', 'precipitation', 'snow_depth']
    for col in weather_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[col], kde=True, bins=20)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.grid(True)
        st.pyplot(fig)

        desc = df[col].describe().round(2).to_frame()
        desc.columns = [f"{col} statistics"]  # 컬럼명 예쁘게 바꾸기
        st.table(desc)

    # 4.3 Counts of Categorical Features
    categorical_cols = ['game_count', 'has_playoff', 'weather_condition', 'is_holiday', 'weekday_final', 'has_game']
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=df[col], order = df[col].value_counts().index) # y for better label readability
        ax.set_title(f'Counts of {col}')
        ax.set_xlabel('Count')
        ax.set_ylabel(col)
        ax.grid(axis='x')
        st.pyplot(fig)

        desc = df[col].describe().round(2).to_frame()
        desc.columns = [f"Value counts for {col}"]  # 컬럼명 예쁘게 바꾸기
        st.table(desc)

    # 4.4 Sport Types Distribution (among days with games)
    sport_counts = {}
    for sport in sorted(list(all_sports)):
        sport_counts[sport] = df[f'is_{sport}'].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    pd.Series(sport_counts).sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title('Number of Days Each Sport Was Played')
    ax.set_ylabel('Number of Days')
    ax.tick_params(axis='x', rotation=45)  # 축 라벨 회전
    ax.grid(axis='y')
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("## 5. 이변량 분석 및 다변량 분석 <span style='font-size:16px; color:gray'>_(Bivariate and Multivariate Analysis)_</span>", unsafe_allow_html=True)

    # 5.1 Accident Count vs. Time Features
    # Monthly
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='month', y='accident_count', data=df)
    ax.set_title('Accident Counts by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Day of the week
    weekday_order = ['월', '화', '수', '목', '금', '토', '일']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='weekday_final', y='accident_count', data=df, order=weekday_order)
    ax.set_title('Accident Counts by Day of the Week')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Year (if multiple years exist)
    if df['year'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='year', y='accident_count', data=df)
        ax.set_title('Accident Counts by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Accident Count')
        ax.grid(True)
        st.pyplot(fig)

    # 5.2 Accident Count vs. Game-Related Features
    # Has Game?
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='has_game', y='accident_count', data=df)
    ax.set_title('Accident Counts: Game Day vs. No Game Day')
    ax.set_xlabel('Was there a game?')
    ax.set_ylabel('Accident Count')
    ax.set_xticks([0, 1])  # tick 위치
    ax.set_xticklabels(['No Game', 'Game'])  # tick 라벨
    ax.grid(True)
    st.pyplot(fig)
    st.write(df.groupby('has_game')['accident_count'].describe())

    # Game Count (for days with games)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='game_count', y='accident_count', data=df[df['has_game']]) # Only for days with games
    ax.set_title('Accident Counts by Number of Games Played (on game days)')
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Has Playoff? (for days with games)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='has_playoff', y='accident_count', data=df[df['has_game']]) # Only for days with games
    ax.set_title('Accident Counts: Playoff Game vs. Regular Game (on game days)')
    ax.set_xlabel('Is it a Playoff Game?')
    ax.set_ylabel('Accident Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Regular Game', 'Playoff Game'])
    ax.grid(True)
    st.pyplot(fig)
    st.write(df[df['has_game']].groupby('has_playoff')['accident_count'].describe())

    # Specific sports
    sports_accidents_mean = {}
    for sport in sorted(list(all_sports)):
        sports_accidents_mean[sport] = df[df[f'is_{sport}']]['accident_count'].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    pd.Series(sports_accidents_mean).sort_values(ascending=False).plot(kind='bar')
    ax.set_title('Average Accident Count on Days When Specific Sport Was Played')
    ax.set_ylabel('Average Accident Count')
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(axis='y')
    ax.axhline(df[df['has_game']==False]['accident_count'].mean(), color='red', linestyle='--', label='Avg Accidents (No Game)')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # 5.3 Accident Count vs. Weather Features
    # Temperature
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='temperature', y='accident_count', data=df, alpha=0.5)
    ax.set_title('Accident Count vs. Temperature')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Precipitation
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='precipitation', y='accident_count', data=df[df['precipitation'] > 0], alpha=0.5)
    ax.set_title('Accident Count vs. Precipitation (for days with precipitation > 0)')
    ax.set_xlabel('Precipitation (mm)')
    ax.set_ylabel('Accident Count')
    ax.set_xscale('log') # Using log scale due to skewness
    ax.grid(True)
    st.pyplot(fig)

    # Snow Depth
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='snow_depth', y='accident_count', data=df[df['snow_depth'] > 0], alpha=0.5)
    ax.set_title('Accident Count vs. Snow Depth (for days with snow_depth > 0)')
    ax.set_xlabel('Snow Depth (cm)')
    ax.set_ylabel('Accident Count')
    ax.set_xscale('log') # Using log scale due to skewness
    ax.grid(True)
    st.pyplot(fig)

    # Weather Condition
    fig, ax = plt.subplots(figsize=(12, 7))
    weather_order = df.groupby('weather_condition')['accident_count'].mean().sort_values(ascending=False).index
    sns.boxplot(x='weather_condition', y='accident_count', data=df, order=weather_order)
    ax.set_title('Accident Counts by Weather Condition')
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Accident Count')
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    st.write(df.groupby('weather_condition')['accident_count'].describe())

    # 5.4 Accident Count vs. Holiday
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='is_holiday', y='accident_count', data=df)
    ax.set_title('Accident Counts: Holiday vs. Non-Holiday')
    ax.set_xlabel('Is it a Holiday?')
    ax.set_ylabel('Accident Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Holiday', 'Holiday'])
    ax.grid(True)
    st.pyplot(fig)
    st.write(df.groupby('is_holiday')['accident_count'].describe())


    # 5.5 Correlation Heatmap
    numerical_cols_for_corr = ['accident_count', 'game_count', 'temperature', 'precipitation', 'snow_depth', 'month', 'day_of_week_num']
    # Add boolean columns (0/1) for correlation
    numerical_cols_for_corr.extend(['has_playoff', 'is_holiday', 'has_game'])
    for sport in sorted(list(all_sports)):
        numerical_cols_for_corr.append(f'is_{sport}')

    correlation_matrix = df[numerical_cols_for_corr].corr()
    fig, ax = plt.subplots(figsize=(18, 15)) # Adjust size as needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    ax.set_title('Correlation Matrix of Numerical and Boolean Features')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("## 6. 사고 건수 시계열 분석 <span style='font-size:16px; color:gray'>_(Time Series Analysis of Accident Counts)_</span>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(18, 6))
    df.set_index('date')['accident_count'].plot()
    ax.set_title('Daily Accident Counts Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Monthly average accident count
    monthly_avg_accidents = df.set_index('date').resample('ME')['accident_count'].mean()
    fig, ax = plt.subplots(figsize=(18, 6))
    monthly_avg_accidents.plot(marker='o')
    ax.set_title('Monthly Average Accident Counts Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("## 7. 상호작용(예시) <span style='font-size:16px; color:gray'>_(Combined Effects (Examples))_</span>", unsafe_allow_html=True)

    # 7.1 Game Day vs. No Game Day by Weather Condition
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='weather_condition', y='accident_count', hue='has_game', data=df, order=weather_order)
    ax.set_title('Accident Counts by Weather Condition, Split by Game Presence')
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Accident Count')
    ax.tick_params(axis='x', rotation=45) 
    ax.legend(title='Game Day')
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

    # 7.2 Playoff vs. Regular Game by Weather (on game days)
    if df['has_game'].any(): # Ensure there are game days
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(x='weather_condition', y='accident_count', hue='has_playoff', 
                    data=df[df['has_game']], order=weather_order)
        ax.set_title('Accident Counts on Game Days by Weather, Split by Playoff Status')
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Accident Count')
        ax.tick_params(axis='x', rotation=45) 
        ax.legend(title='Playoff Game')
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

    # 7.3 Precipitation effect on game days vs non-game days
    df['has_precipitation'] = df['precipitation'] > 0
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='has_game', y='accident_count', hue='has_precipitation', data=df)
    ax.set_title('Impact of Precipitation on Accident Counts (Game vs. No Game)')
    ax.set_xlabel('Was there a game?')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Game', 'Game Day'])
    ax.set_ylabel('Average Accident Count')
    ax.legend(title='Has Precipitation')
    ax.grid(axis='y')
    st.pyplot(fig)

    # 7.4 Snow effect on game days vs non-game days
    df['has_snow'] = df['snow_depth'] > 0
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='has_game', y='accident_count', hue='has_snow', data=df)
    ax.set_title('Impact of Snow on Accident Counts (Game vs. No Game)')
    ax.set_xlabel('Was there a game?')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Game', 'Game Day'])
    ax.set_ylabel('Average Accident Count')
    ax.legend(title='Has Snow')
    ax.grid(axis='y')
    st.pyplot(fig)

    st.markdown("## 8. 극단적 사고 중점 결과값 <span style='font-size:16px; color:gray'>_(Focus on Extreme Accident Days)_</span>", unsafe_allow_html=True)

    # What are the conditions on days with very high accident counts?
    high_accident_threshold = df['accident_count'].quantile(0.90) # Top 10%
    high_accident_days = df[df['accident_count'] >= high_accident_threshold]
    st.write(f"\nDays with accident_count >= {high_accident_threshold} (Top 10%):\n")
    st.write(high_accident_days[['date', 'accident_count', 'game_count', 'sports_type', 'has_playoff', 'temperature', 'precipitation', 'snow_depth', 'weather_condition', 'is_holiday', 'weekday_final']].sort_values(by='accident_count', ascending=False))

    st.write("\nWeather conditions on high accident days:")
    st.write(high_accident_days['weather_condition'].value_counts(normalize=True)*100)

    st.write("\nGame presence on high accident days:")
    st.write(high_accident_days['has_game'].value_counts(normalize=True)*100)

    st.write("\nPlayoff presence on high accident (game) days:")
    if high_accident_days['has_game'].any():
        st.write(high_accident_days[high_accident_days['has_game']]['has_playoff'].value_counts(normalize=True)*100)

with tabs[1]:
    df = pd.read_csv('./04.data_preprocess/2nd-dataset_경기 수원시_20250602_141514.csv')
    
    st.write(df.columns)

    # Set Korean font for matplotlib
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False # To display minus sign correctly

    st.markdown("## 1. 초기 데이터 검사 <span style='font-size:16px; color:gray'>_(Initial Data Inspection)_</span>", unsafe_allow_html=True)

    st.write("Shape of the dataframe:", df.shape)
    st.write("\nFirst 5 rows:")
    st.write(df.head())
    st.write("\nData types and non-null counts:")
    df.info()
    st.write("\nSummary statistics for numerical columns:")
    st.write(df.describe())
    st.write("\nSummary statistics for object columns:")
    st.write(df.describe(include='object'))
    st.write("\nMissing values per column:")
    st.write(df.isnull().sum())


    st.markdown("## 2. 데이터 정제 및 전처리 <span style='font-size:16px; color:gray'>_(Data Cleaning and Preprocessing)_</span>", unsafe_allow_html=True)

    # Convert 'date' to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # Convert boolean-like columns to actual booleans for clarity if needed (0/1 is fine for most analyses)
    df['has_playoff'] = df['has_playoff'].astype(bool)
    df['is_holiday'] = df['is_holiday'].astype(bool)

    # Create a column 'has_game' for easier filtering
    df['has_game'] = df['game_count'] > 0

    # Check consistency: region_code
    st.write("\nUnique values in 'region_code':", df['region_code'].unique())
    # If 'region_code' is constant, we might drop it for modeling, but keep for EDA context.

    # Clean sports_type: "없음" means no sport, otherwise it's a list
    # For analysis, we can create dummy variables for common sports
    all_sports = set()
    for sports_list in df['sports_type'].unique():
        if sports_list != "없음":
            for sport in sports_list.split(','):
                all_sports.add(sport.strip()) # .strip() to remove potential leading/trailing spaces

    all_sports_sorted = sorted(list(all_sports))
    markdown_list = "\n".join([f"{i+1}. {sport}" for i, sport in enumerate(all_sports_sorted)])
    st.markdown(f"**Identified unique sport types:**\n\n{markdown_list}")


    # Create boolean columns for each identified sport
    for sport in sorted(list(all_sports)):
        df[f'is_{sport}'] = df['sports_type'].apply(lambda x: sport in x)

    st.write("\nDataFrame with new sport boolean columns (first 5 rows):")
    st.write(df.head())


    st.markdown("## 3. 피처 엔지니어링 <span style='font-size:16px; color:gray'>_(Feature Engineering (Time-based features))_</span>", unsafe_allow_html=True)

    # 1️⃣ date 컬럼을 datetime 형식으로 변환
    df['date'] = pd.to_datetime(df['date'])

    # 2️⃣ datetime 관련 파생 변수 생성
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week_num'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['day_name'] = df['date'].dt.day_name()

    # 3️⃣ 요일 이름 매핑 (한글)
    day_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
    df['derived_weekday'] = df['day_of_week_num'].map(day_map)

    # 4️⃣ provided weekday와 derived weekday 비교
    if 'weekday' in df.columns:
        mismatch_count = (df['derived_weekday'] != df['weekday']).sum()
        st.write(f"\nNumber of mismatches between derived weekday and provided weekday: {mismatch_count}")
        if mismatch_count > 0:
            st.write("Consider using the derived weekday or thoroughly checking the provided one.")
            df['weekday_final'] = df['derived_weekday']
        else:
            df['weekday_final'] = df['weekday']
    else:
        df['weekday_final'] = df['derived_weekday']



    st.markdown("## 4. 단변량 분석 <span style='font-size:16px; color:gray'>_(Univariate Analysis)_</span>", unsafe_allow_html=True)

    # 4.1 Distribution of Accident Count
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['accident_count'], kde=True, bins=20)
    ax.set_title('수원 일일 사고 발생 건수 분포도', fontsize=16)
    ax.set_xlabel('사고 건수')
    ax.set_ylabel('사고 빈도')
    ax.grid(True)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df['accident_count'])
    ax.set_title('수원 일일 사고 건수 박스 플롯', fontsize=16)
    ax.set_xlabel('사고 건수')
    ax.grid(True)
    st.pyplot(fig)
        
    desc = df['accident_count'].describe().round(2).to_frame()
    desc.columns = ['Accident count statistics']  # 컬럼명 예쁘게 바꾸기
    st.table(desc)

    # 4.2 Distribution of Numerical Weather Features
    weather_cols = ['temperature', 'precipitation', 'snow_depth']
    for col in weather_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[col], kde=True, bins=20)
        ax.set_title(f'{col} 분포', fontsize=16)
        ax.set_xlabel(col)
        ax.set_ylabel('사고 빈도')
        ax.grid(True)
        st.pyplot(fig)
            
        desc = df[col].describe().round(2).to_frame()
        desc.columns = [f"{col} statistics"]  # 컬럼명 예쁘게 바꾸기
        st.table(desc)

    # 4.3 Counts of Categorical Features
    categorical_cols = ['game_count', 'has_playoff', 'weather_condition', 'is_holiday', 'weekday_final', 'has_game']
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=df[col], order = df[col].value_counts().index) # y for better label readability
        ax.set_title(f'Counts of {col}')
        ax.set_xlabel('Count')
        ax.set_ylabel(col)
        ax.grid(axis='x')
        st.pyplot(fig)

        desc = df[col].describe().round(2).to_frame()
        desc.columns = [f"\nValue counts for {col}"]  # 컬럼명 예쁘게 바꾸기
        st.table(desc)

    # 4.4 Sport Types Distribution (among days with games)
    sport_counts = {}
    for sport in sorted(list(all_sports)):
        sport_counts[sport] = df[f'is_{sport}'].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    pd.Series(sport_counts).sort_values(ascending=False).plot(kind='bar')
    ax.set_title('Number of Days Each Sport Was Played')
    ax.set_ylabel('Number of Days')
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(axis='y')
    fig.tight_layout()
    st.pyplot(fig)


    st.markdown("## 5. 이변량 분석 및 다변량 분석 <span style='font-size:16px; color:gray'>_(Bivariate and Multivariate Analysis)_</span>", unsafe_allow_html=True)

    # 1️⃣ date 컬럼을 datetime 형식으로 변환
    df['date'] = pd.to_datetime(df['date'])

    # 2️⃣ datetime 관련 파생 변수 생성
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week_num'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['day_name'] = df['date'].dt.day_name()

    # 5.1 Accident Count vs. Time Features
    # Monthly
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='month', y='accident_count', data=df)
    ax.set_title('Accident Counts by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Day of the week
    weekday_order = ['월', '화', '수', '목', '금', '토', '일']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='weekday_final', y='accident_count', data=df, order=weekday_order)
    ax.set_title('Accident Counts by Day of the Week')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Year (if multiple years exist)
    if df['year'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='year', y='accident_count', data=df)
        ax.set_title('Accident Counts by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Accident Count')
        ax.grid(True)
        st.pyplot(fig)

    # 5.2 Accident Count vs. Game-Related Features
    # Has Game?
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='has_game', y='accident_count', data=df)
    ax.set_title('Accident Counts: Game Day vs. No Game Day')
    ax.set_xlabel('Was there a game?')
    ax.set_ylabel('Accident Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Game', 'Game'])
    ax.grid(True)
    st.pyplot(fig)
    st.write(df.groupby('has_game')['accident_count'].describe())

    # Game Count (for days with games)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='game_count', y='accident_count', data=df[df['has_game']]) # Only for days with games
    ax.set_title('Accident Counts by Number of Games Played (on game days)')
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Has Playoff? (for days with games)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='has_playoff', y='accident_count', data=df[df['has_game']]) # Only for days with games
    ax.set_title('Accident Counts: Playoff Game vs. Regular Game (on game days)')
    ax.set_xlabel('Is it a Playoff Game?')
    ax.set_ylabel('Accident Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Regular Game', 'Playoff Game'])
    ax.grid(True)
    st.pyplot(fig)
    st.write(df[df['has_game']].groupby('has_playoff')['accident_count'].describe())

    # Specific sports
    sports_accidents_mean = {}
    for sport in sorted(list(all_sports)):
        sports_accidents_mean[sport] = df[df[f'is_{sport}']]['accident_count'].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    pd.Series(sports_accidents_mean).sort_values(ascending=False).plot(kind='bar')
    ax.set_title('Average Accident Count on Days When Specific Sport Was Played')
    ax.set_ylabel('Average Accident Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    ax.axhline(df[df['has_game']==False]['accident_count'].mean(), color='red', linestyle='--', label='Avg Accidents (No Game)')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # 5.3 Accident Count vs. Weather Features
    # Temperature
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='temperature', y='accident_count', data=df, alpha=0.5)
    ax.set_title('Accident Count vs. Temperature')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Precipitation
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='precipitation', y='accident_count', data=df[df['precipitation'] > 0], alpha=0.5)
    ax.set_title('Accident Count vs. Precipitation (for days with precipitation > 0)')
    ax.set_xlabel('Precipitation (mm)')
    ax.set_ylabel('Accident Count')
    ax.set_xscale('log') # Using log scale due to skewness
    ax.grid(True)
    st.pyplot(fig)

    # Snow Depth
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='snow_depth', y='accident_count', data=df[df['snow_depth'] > 0], alpha=0.5)
    ax.set_title('Accident Count vs. Snow Depth (for days with snow_depth > 0)')
    ax.set_xlabel('Snow Depth (cm)')
    ax.set_ylabel('Accident Count')
    ax.set_xscale('log') # Using log scale due to skewness
    ax.grid(True)
    st.pyplot(fig)

    # Weather Condition
    fig, ax = plt.subplots(figsize=(12, 7))
    weather_order = df.groupby('weather_condition')['accident_count'].mean().sort_values(ascending=False).index
    sns.boxplot(x='weather_condition', y='accident_count', data=df, order=weather_order)
    ax.set_title('Accident Counts by Weather Condition')
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Accident Count')
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    st.write(df.groupby('weather_condition')['accident_count'].describe())

    # 5.4 Accident Count vs. Holiday
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='is_holiday', y='accident_count', data=df)
    ax.set_title('Accident Counts: Holiday vs. Non-Holiday')
    ax.set_xlabel('Is it a Holiday?')
    ax.set_ylabel('Accident Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Holiday', 'Holiday'])
    ax.grid(True)
    st.pyplot(fig)
    st.write(df.groupby('is_holiday')['accident_count'].describe())


    # 5.5 Correlation Heatmap
    numerical_cols_for_corr = ['accident_count', 'game_count', 'temperature', 'precipitation', 'snow_depth', 'month', 'day_of_week_num']
    # Add boolean columns (0/1) for correlation
    numerical_cols_for_corr.extend(['has_playoff', 'is_holiday', 'has_game'])
    for sport in sorted(list(all_sports)):
        numerical_cols_for_corr.append(f'is_{sport}')

    correlation_matrix = df[numerical_cols_for_corr].corr()
    fig, ax = plt.subplots(figsize=(18, 15)) # Adjust size as needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    ax.set_title('Correlation Matrix of Numerical and Boolean Features')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    st.pyplot(fig)


    st.markdown("## 6. 사고 건수 시계열 분석 <span style='font-size:16px; color:gray'>_(Time Series Analysis of Accident Counts)_</span>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(18, 6))
    df.set_index('date')['accident_count'].plot()
    ax.set_title('Daily Accident Counts Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Monthly average accident count
    monthly_avg_accidents = df.set_index('date').resample('M')['accident_count'].mean()
    fig, ax = plt.subplots(figsize=(18, 6))
    monthly_avg_accidents.plot(marker='o')
    ax.set_title('Monthly Average Accident Counts Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Accident Count')
    ax.grid(True)
    st.pyplot(fig)



    st.markdown("## 7. 상호작용(예시) <span style='font-size:16px; color:gray'>_(Combined Effects (Examples))_</span>", unsafe_allow_html=True)

    # 7.1 Game Day vs. No Game Day by Weather Condition
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='weather_condition', y='accident_count', hue='has_game', data=df, order=weather_order)
    ax.set_title('Accident Counts by Weather Condition, Split by Game Presence')
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Accident Count')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Game Day')
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

    # 7.2 Playoff vs. Regular Game by Weather (on game days)
    if df['has_game'].any(): # Ensure there are game days
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(x='weather_condition', y='accident_count', hue='has_playoff', 
                    data=df[df['has_game']], order=weather_order)
        ax.set_title('Accident Counts on Game Days by Weather, Split by Playoff Status')
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Accident Count')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Playoff Game')
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

    # 7.3 Precipitation effect on game days vs non-game days
    df['has_precipitation'] = df['precipitation'] > 0
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='has_game', y='accident_count', hue='has_precipitation', data=df)
    ax.set_title('Impact of Precipitation on Accident Counts (Game vs. No Game)')
    ax.set_xlabel('Was there a game?')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Game', 'Game Day'])
    ax.set_ylabel('Average Accident Count')
    ax.legend(title='Has Precipitation')
    ax.grid(axis='y')
    st.pyplot(fig)

    # 7.4 Snow effect on game days vs non-game days
    df['has_snow'] = df['snow_depth'] > 0
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='has_game', y='accident_count', hue='has_snow', data=df)
    ax.set_title('Impact of Snow on Accident Counts (Game vs. No Game)')
    ax.set_xlabel('Was there a game?')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Game', 'Game Day'])
    ax.set_ylabel('Average Accident Count')
    ax.legend(title='Has Snow')
    ax.grid(axis='y')
    st.pyplot(fig)


    st.markdown("## 8. 극단적 사고 중점 결과값 <span style='font-size:16px; color:gray'>_(Focus on Extreme Accident Days)_</span>", unsafe_allow_html=True)

    # What are the conditions on days with very high accident counts?
    high_accident_threshold = df['accident_count'].quantile(0.90) # Top 10%
    high_accident_days = df[df['accident_count'] >= high_accident_threshold]
    st.write(f"\nDays with accident_count >= {high_accident_threshold} (Top 10%):\n")
    st.write(high_accident_days[['date', 'accident_count', 'game_count', 'sports_type', 'has_playoff', 'temperature', 'precipitation', 'snow_depth', 'weather_condition', 'is_holiday', 'weekday_final']].sort_values(by='accident_count', ascending=False))

    st.write("\nWeather conditions on high accident days:")
    st.write(high_accident_days['weather_condition'].value_counts(normalize=True)*100)

    st.write("\nGame presence on high accident days:")
    st.write(high_accident_days['has_game'].value_counts(normalize=True)*100)

    st.write("\nPlayoff presence on high accident (game) days:")
    if high_accident_days['has_game'].any():
        st.write(high_accident_days[high_accident_days['has_game']]['has_playoff'].value_counts(normalize=True)*100)

  
with tabs[2]:
    df2 = pd.read_csv('./04.data_preprocess/2nd-dataset_서울 송파구_20250602_110023.csv')

    # Set Korean font for matplotlib
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False # To display minus sign correctly

    st.markdown("## 1. 초기 데이터 검사 <span style='font-size:16px; color:gray'>_(Initial Data Inspection)_</span>", unsafe_allow_html=True)

    st.write("Shape of the dataframe:", df2.shape)
    st.write("\nFirst 5 rows:")
    st.write(df2.head())
    st.write("\nData types and non-null counts:")
    df2.info()
    st.write("\nSummary statistics for numerical columns:")
    st.write(df2.describe())
    st.write("\nSummary statistics for object columns:")
    st.write(df2.describe(include='object'))
    st.write("\nMissing values per column:")
    st.write(df2.isnull().sum())


    st.markdown("## 2. 데이터 정제 및 전처리 <span style='font-size:16px; color:gray'>_(Data Cleaning and Preprocessing)_</span>", unsafe_allow_html=True)

    # Convert 'date' to datetime objects
    df2['date'] = pd.to_datetime(df2['date'])

    # Convert boolean-like columns to actual booleans for clarity if needed (0/1 is fine for most analyses)
    df2['is_post_season'] = df2['is_post_season'].astype(bool)
    df2['is_holiday'] = df2['is_holiday'].astype(bool)

    # Create a column 'has_game' for easier filtering
    df2['has_game'] = df2['game_count'] > 0

    # Check consistency: region_code
    st.write("\nUnique values in 'region_code':", df2['region_code'].unique())
    # If 'region_code' is constant, we might drop it for modeling, but keep for EDA context.

    # Clean sports_types: "없음" means no sport, otherwise it's a list
    # For analysis, we can create dummy variables for common sports
    all_sports = set()
    for sports_list in df2['sports_types'].unique():
        if sports_list != "없음":
            for sport in sports_list.split(','):
                all_sports.add(sport.strip()) # .strip() to remove potential leading/trailing spaces

    all_sports_sorted = sorted(list(all_sports))
    markdown_list = "\n".join([f"{i+1}. {sport}" for i, sport in enumerate(all_sports_sorted)])
    st.markdown(f"**Identified unique sport types:**\n\n{markdown_list}")


    # Create boolean columns for each identified sport
    for sport in sorted(list(all_sports)):
        df2[f'is_{sport}'] = df2['sports_types'].apply(lambda x: sport in x)

    st.write("\nDataFrame with new sport boolean columns (first 5 rows):")
    st.write(df2.head())


    st.markdown("## 3. 피처 엔지니어링 <span style='font-size:16px; color:gray'>_(Feature Engineering (Time-based features))_</span>", unsafe_allow_html=True)

    # 1️⃣ date 컬럼을 datetime 형식으로 변환
    df2['date'] = pd.to_datetime(df2['date'])

    # 2️⃣ datetime 관련 파생 변수 생성
    df2['year'] = df2['date'].dt.year
    df2['month'] = df2['date'].dt.month
    df2['day_of_week_num'] = df2['date'].dt.dayofweek  # Monday=0, Sunday=6
    df2['day_name'] = df2['date'].dt.day_name()

    # 3️⃣ 요일 이름 매핑 (한글)
    day_map = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
    df2['derived_weekday'] = df2['day_of_week_num'].map(day_map)

    # 4️⃣ provided weekday와 derived weekday 비교
    if 'weekday' in df2.columns:
        mismatch_count = (df2['derived_weekday'] != df2['weekday']).sum()
        st.write(f"\nNumber of mismatches between derived weekday and provided weekday: {mismatch_count}")
        if mismatch_count > 0:
            st.write("Consider using the derived weekday or thoroughly checking the provided one.")
            df2['weekday_final'] = df2['derived_weekday']
        else:
            df2['weekday_final'] = df2['weekday']
    else:
        df2['weekday_final'] = df2['derived_weekday']



    st.markdown("## 4. 단변량 분석 <span style='font-size:16px; color:gray'>_(Univariate Analysis)_</span>", unsafe_allow_html=True)

    # 4.1 Distribution of Accident Count
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df2['accident_count'], kde=True, bins=20)
    ax.set_title('수원 일일 사고 발생 건수 분포도', fontsize=16)
    ax.set_xlabel('사고 건수')
    ax.set_ylabel('사고 빈도')
    ax.grid(True)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df2['accident_count'])
    ax.set_title('수원 일일 사고 건수 박스 플롯', fontsize=16)
    ax.set_xlabel('사고 건수')
    ax.grid(True)
    st.pyplot(fig)
        
    desc = df2['accident_count'].describe().round(2).to_frame()
    desc.columns = ['Accident count statistics']  # 컬럼명 예쁘게 바꾸기
    st.table(desc)

    # 4.2 Distribution of Numerical Weather Features
    weather_cols = ['temperature', 'precipitation', 'snow_depth']
    for col in weather_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df2[col], kde=True, bins=20)
        ax.set_title(f'{col} 분포', fontsize=16)
        ax.set_xlabel(col)
        ax.set_ylabel('사고 빈도')
        ax.grid(True)
        st.pyplot(fig)
            
        desc = df2[col].describe().round(2).to_frame()
        desc.columns = [f"{col} statistics"]  # 컬럼명 예쁘게 바꾸기
        st.table(desc)

    # 4.3 Counts of Categorical Features
    categorical_cols = ['game_count', 'is_post_season', 'weather_condition', 'is_holiday', 'weekday_final', 'has_game']
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(y=df2[col], order = df2[col].value_counts().index) # y for better label readability
        ax.set_title(f'Counts of {col}')
        ax.set_xlabel('Count')
        ax.set_ylabel(col)
        ax.grid(axis='x')
        st.pyplot(fig)

        desc = df2[col].describe().round(2).to_frame()
        desc.columns = [f"\nValue counts for {col}"]  # 컬럼명 예쁘게 바꾸기
        st.table(desc)

    # 4.4 Sport Types Distribution (among days with games)
    sport_counts = {}
    for sport in sorted(list(all_sports)):
        sport_counts[sport] = df2[f'is_{sport}'].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    pd.Series(sport_counts).sort_values(ascending=False).plot(kind='bar')
    ax.set_title('Number of Days Each Sport Was Played')
    ax.set_ylabel('Number of Days')
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(axis='y')
    fig.tight_layout()
    st.pyplot(fig)


    st.markdown("## 5. 이변량 분석 및 다변량 분석 <span style='font-size:16px; color:gray'>_(Bivariate and Multivariate Analysis)_</span>", unsafe_allow_html=True)

    # 1️⃣ date 컬럼을 datetime 형식으로 변환
    df2['date'] = pd.to_datetime(df2['date'])

    # 2️⃣ datetime 관련 파생 변수 생성
    df2['year'] = df2['date'].dt.year
    df2['month'] = df2['date'].dt.month
    df2['day_of_week_num'] = df2['date'].dt.dayofweek  # Monday=0, Sunday=6
    df2['day_name'] = df2['date'].dt.day_name()

    # 5.1 Accident Count vs. Time Features
    # Monthly
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='month', y='accident_count', data=df2)
    ax.set_title('Accident Counts by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Day of the week
    weekday_order = ['월', '화', '수', '목', '금', '토', '일']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='weekday_final', y='accident_count', data=df2, order=weekday_order)
    ax.set_title('Accident Counts by Day of the Week')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Year (if multiple years exist)
    if df2['year'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='year', y='accident_count', data=df2)
        ax.set_title('Accident Counts by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Accident Count')
        ax.grid(True)
        st.pyplot(fig)

    # 5.2 Accident Count vs. Game-Related Features
    # Has Game?
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='has_game', y='accident_count', data=df2)
    ax.set_title('Accident Counts: Game Day vs. No Game Day')
    ax.set_xlabel('Was there a game?')
    ax.set_ylabel('Accident Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Game', 'Game'])
    ax.grid(True)
    st.pyplot(fig)
    st.write(df2.groupby('has_game')['accident_count'].describe())

    # Game Count (for days with games)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='game_count', y='accident_count', data=df2[df2['has_game']]) # Only for days with games
    ax.set_title('Accident Counts by Number of Games Played (on game days)')
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Has Playoff? (for days with games)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='is_post_season', y='accident_count', data=df2[df2['has_game']]) # Only for days with games
    ax.set_title('Accident Counts: Playoff Game vs. Regular Game (on game days)')
    ax.set_xlabel('Is it a Playoff Game?')
    ax.set_ylabel('Accident Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Regular Game', 'Playoff Game'])
    ax.grid(True)
    st.pyplot(fig)
    st.write(df2[df2['has_game']].groupby('is_post_season')['accident_count'].describe())

    # Specific sports
    sports_accidents_mean = {}
    for sport in sorted(list(all_sports)):
        sports_accidents_mean[sport] = df2[df2[f'is_{sport}']]['accident_count'].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    pd.Series(sports_accidents_mean).sort_values(ascending=False).plot(kind='bar')
    ax.set_title('Average Accident Count on Days When Specific Sport Was Played')
    ax.set_ylabel('Average Accident Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    ax.axhline(df2[df2['has_game']==False]['accident_count'].mean(), color='red', linestyle='--', label='Avg Accidents (No Game)')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # 5.3 Accident Count vs. Weather Features
    # Temperature
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='temperature', y='accident_count', data=df2, alpha=0.5)
    ax.set_title('Accident Count vs. Temperature')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Precipitation
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='precipitation', y='accident_count', data=df2[df2['precipitation'] > 0], alpha=0.5)
    ax.set_title('Accident Count vs. Precipitation (for days with precipitation > 0)')
    ax.set_xlabel('Precipitation (mm)')
    ax.set_ylabel('Accident Count')
    ax.set_xscale('log') # Using log scale due to skewness
    ax.grid(True)
    st.pyplot(fig)

    # Snow Depth
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='snow_depth', y='accident_count', data=df2[df2['snow_depth'] > 0], alpha=0.5)
    ax.set_title('Accident Count vs. Snow Depth (for days with snow_depth > 0)')
    ax.set_xlabel('Snow Depth (cm)')
    ax.set_ylabel('Accident Count')
    ax.set_xscale('log') # Using log scale due to skewness
    ax.grid(True)
    st.pyplot(fig)

    # Weather Condition
    fig, ax = plt.subplots(figsize=(12, 7))
    weather_order = df2.groupby('weather_condition')['accident_count'].mean().sort_values(ascending=False).index
    sns.boxplot(x='weather_condition', y='accident_count', data=df2, order=weather_order)
    ax.set_title('Accident Counts by Weather Condition')
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Accident Count')
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    st.write(df2.groupby('weather_condition')['accident_count'].describe())

    # 5.4 Accident Count vs. Holiday
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='is_holiday', y='accident_count', data=df2)
    ax.set_title('Accident Counts: Holiday vs. Non-Holiday')
    ax.set_xlabel('Is it a Holiday?')
    ax.set_ylabel('Accident Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Holiday', 'Holiday'])
    ax.grid(True)
    st.pyplot(fig)
    st.write(df2.groupby('is_holiday')['accident_count'].describe())


    # 5.5 Correlation Heatmap
    numerical_cols_for_corr = ['accident_count', 'game_count', 'temperature', 'precipitation', 'snow_depth', 'month', 'day_of_week_num']
    # Add boolean columns (0/1) for correlation
    numerical_cols_for_corr.extend(['is_post_season', 'is_holiday', 'has_game'])
    for sport in sorted(list(all_sports)):
        numerical_cols_for_corr.append(f'is_{sport}')

    correlation_matrix = df2[numerical_cols_for_corr].corr()
    fig, ax = plt.subplots(figsize=(18, 15)) # Adjust size as needed
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    ax.set_title('Correlation Matrix of Numerical and Boolean Features')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    st.pyplot(fig)


    st.markdown("## 6. 사고 건수 시계열 분석 <span style='font-size:16px; color:gray'>_(Time Series Analysis of Accident Counts)_</span>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(18, 6))
    df2.set_index('date')['accident_count'].plot()
    ax.set_title('Daily Accident Counts Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Accident Count')
    ax.grid(True)
    st.pyplot(fig)

    # Monthly average accident count
    monthly_avg_accidents = df2.set_index('date').resample('M')['accident_count'].mean()
    fig, ax = plt.subplots(figsize=(18, 6))
    monthly_avg_accidents.plot(marker='o')
    ax.set_title('Monthly Average Accident Counts Over Time')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Accident Count')
    ax.grid(True)
    st.pyplot(fig)



    st.markdown("## 7. 상호작용(예시) <span style='font-size:16px; color:gray'>_(Combined Effects (Examples))_</span>", unsafe_allow_html=True)

    # 7.1 Game Day vs. No Game Day by Weather Condition
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='weather_condition', y='accident_count', hue='has_game', data=df2, order=weather_order)
    ax.set_title('Accident Counts by Weather Condition, Split by Game Presence')
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Accident Count')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Game Day')
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

    # 7.2 Playoff vs. Regular Game by Weather (on game days)
    if df2['has_game'].any(): # Ensure there are game days
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(x='weather_condition', y='accident_count', hue='is_post_season', 
                    data=df2[df2['has_game']], order=weather_order)
        ax.set_title('Accident Counts on Game Days by Weather, Split by Playoff Status')
        ax.set_xlabel('Weather Condition')
        ax.set_ylabel('Accident Count')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Playoff Game')
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

    # 7.3 Precipitation effect on game days vs non-game days
    df2['has_precipitation'] = df2['precipitation'] > 0
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='has_game', y='accident_count', hue='has_precipitation', data=df2)
    ax.set_title('Impact of Precipitation on Accident Counts (Game vs. No Game)')
    ax.set_xlabel('Was there a game?')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Game', 'Game Day'])
    ax.set_ylabel('Average Accident Count')
    ax.legend(title='Has Precipitation')
    ax.grid(axis='y')
    st.pyplot(fig)

    # 7.4 Snow effect on game days vs non-game days
    df2['has_snow'] = df2['snow_depth'] > 0
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='has_game', y='accident_count', hue='has_snow', data=df2)
    ax.set_title('Impact of Snow on Accident Counts (Game vs. No Game)')
    ax.set_xlabel('Was there a game?')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Game', 'Game Day'])
    ax.set_ylabel('Average Accident Count')
    ax.legend(title='Has Snow')
    ax.grid(axis='y')
    st.pyplot(fig)


    st.markdown("## 8. 극단적 사고 중점 결과값 <span style='font-size:16px; color:gray'>_(Focus on Extreme Accident Days)_</span>", unsafe_allow_html=True)

    # What are the conditions on days with very high accident counts?
    high_accident_threshold = df2['accident_count'].quantile(0.90) # Top 10%
    high_accident_days = df2[df2['accident_count'] >= high_accident_threshold]
    st.write(f"\nDays with accident_count >= {high_accident_threshold} (Top 10%):\n")
    st.write(high_accident_days[['date', 'accident_count', 'game_count', 'sports_types', 'is_post_season', 'temperature', 'precipitation', 'snow_depth', 'weather_condition', 'is_holiday', 'weekday_final']].sort_values(by='accident_count', ascending=False))

    st.write("\nWeather conditions on high accident days:")
    st.write(high_accident_days['weather_condition'].value_counts(normalize=True)*100)

    st.write("\nGame presence on high accident days:")
    st.write(high_accident_days['has_game'].value_counts(normalize=True)*100)

    st.write("\nPlayoff presence on high accident (game) days:")
    if high_accident_days['has_game'].any():
        st.write(high_accident_days[high_accident_days['has_game']]['is_post_season'].value_counts(normalize=True)*100)