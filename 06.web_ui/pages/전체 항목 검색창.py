# 1. 데이터베이스에서 데이터 로드 (처음 한번만 실행)

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date, timedelta, time 
import json
import numpy as np
import streamlit as st
import io

st.title("전체 항목 검색창")
st.write("")

with open("./db_config.json", "r") as f:
    config = json.load(f)
    
DB_USER = config["DB_USER"]
DB_PASSWORD = config["DB_PASSWORD"]
DB_HOST = config["DB_HOST"]
DB_NAME = config["DB_NAME"]
DB_PORT = config["DB_PORT"]

# SQLAlchemy 엔진 생성
engine_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

try:
    engine = create_engine(engine_url)
    st.success("**MySQL 데이터베이스에 성공적으로 연결되었습니다.**")
except Exception as e:
    st.error(f"**데이터베이스 연결 오류: {e}**")
    exit()

# --- 데이터베이스에서 데이터 로드하는 함수 ---
def load_table_to_df(table_name, engine):
    """지정된 테이블에서 모든 데이터를 Pandas DataFrame으로 로드합니다."""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
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
        st.error(f"**'{table_name}' 테이블 로드 중 오류 발생: {e}**")
        return pd.DataFrame() # 오류 발생 시 빈 DataFrame 반환

# --- 데이터 로드 ---
stadium_df = load_table_to_df('stadium', engine)
sports_game_df = load_table_to_df('sports_game', engine)
traffic_accident_df = load_table_to_df('traffic_accident', engine)
weather_df = load_table_to_df('weather', engine)

# 데이터베이스 연결 종료
if 'engine' in locals() and engine:
    engine.dispose()
    st.write("**데이터베이스 연결이 종료되었습니다.**")

# 2. 데이터셋을 구성할 지역구 입력

START_DATE = pd.to_datetime("20230101") # 분석 시작일
END_DATE = pd.to_datetime("20241231")   # 분석 종료일
stadium_region_list = stadium_df['region'].unique().tolist()
sports_list = ['야구','축구','배구','농구']
weekday_list = ['월','화','수','목','금','토','일']
TARGET_REGION = []
TARGET_SPORTS = []
TARGET_WEEKDAYS = []
TARGET_DATES = []

# Sample Data
stadium_df = pd.DataFrame({'region': ['서울 송파구', '서울 마포구', '부산 해운대구']})

# Step 변수 초기화
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'TARGET_REGION' not in st.session_state:
    st.session_state.TARGET_REGION = []
if 'TARGET_SPORTS' not in st.session_state:
    st.session_state.TARGET_SPORTS = []
if 'TARGET_WEEKDAYS' not in st.session_state:
    st.session_state.TARGET_WEEKDAYS = []
if 'TARGET_DATES' not in st.session_state:
    st.session_state.TARGET_DATES = []
if 'START_DATE' not in st.session_state:
    st.session_state.START_DATE = pd.to_datetime("20230101") # 분석 시작일
if 'END_DATE' not in st.session_state:
    st.session_state.END_DATE = pd.to_datetime("20241231")   # 분석 종료일
if 'stadium_region_list' not in st.session_state:
    st.session_state.stadium_region_list = stadium_df['region'].unique().tolist()
if 'sports_list' not in st.session_state:
    st.session_state.sports_list = ['야구','축구','배구','농구']
if 'weekday_list' not in st.session_state:
    st.session_state.weekday_list = ['월','화','수','목','금','토','일']
if 'stadium_df' not in st.session_state:
    st.session_state.stadium_df = load_table_to_df('stadium', engine)
if 'sports_game_df' not in st.session_state:
    st.session_state.sports_game_df = load_table_to_df('sports_game', engine)
if 'traffic_accident_df' not in st.session_state:
    st.session_state.traffic_accident_df = load_table_to_df('traffic_accident', engine)
if 'weather_df' not in st.session_state:
    st.session_state.weather_df = load_table_to_df('weather', engine)

# 단계별 입력
if st.session_state.step == 1:
    st.header("Step 1: 지역구 입력")
    input_region = st.text_input("**데이터셋을 구성할 지역구를 입력해주세요 (예시:송파구,마포구)**")
    if st.button("다음", key="step1"):
        if not input_region or input_region.strip() == '전체':
            st.session_state.TARGET_REGION = stadium_region_list
            st.session_state.step = 3
        else:
            input_region_list = [i.strip() for i in input_region.split(',')]
            for iv in input_region_list:
                for item in stadium_region_list:
                    if iv in item:
                        st.session_state.TARGET_REGION.append(item)
            if st.session_state.TARGET_REGION:
                st.session_state.step = 3
            else:
                st.warning(f"**{input_region}에 해당하는 지역이 없습니다. 다시 입력해주세요.**")

elif st.session_state.step == 2:
    st.header("Step 2: 스포츠 종목 입력")
    input_sports = st.text_input("데이터셋을 구성할 스포츠를 입력해주세요 (예시:야구,배구)")
    if st.button("다음", key="step2"):
        if not input_sports or input_sports.strip() == '전체':
            st.session_state.TARGET_SPORTS = sports_list
            st.session_state.step = 3
        else:
            input_sports_list = [i.strip() for i in input_sports.split(',')]
            for iv in input_sports_list:
                for item in sports_list:
                    if iv in item:
                        st.session_state.TARGET_SPORTS.append(item)
            if st.session_state.TARGET_SPORTS:
                st.session_state.step = 3
            else:
                st.warning(f"{input_sports}에 해당하는 종목이 없습니다. 다시 입력해주세요.")

elif st.session_state.step == 3:
    st.header("Step 3: 요일 입력")
    input_weekday = st.text_input("데이터셋을 구성할 요일을 입력해주세요 (예시:월,화,수)")
    if st.button("다음", key="step3"):
        if not input_weekday or input_weekday.strip() == '전체':
            st.session_state.TARGET_WEEKDAYS = weekday_list
            st.session_state.step = 4
        else:
            input_weekday_list = [i.strip() for i in input_weekday.split(',')]
            for iv in input_weekday_list:
                for item in weekday_list:
                    if iv in item:
                        st.session_state.TARGET_WEEKDAYS.append(item)
            if st.session_state.TARGET_WEEKDAYS:
                st.session_state.step = 4
            else:
                st.warning(f"{input_weekday}에 해당하는 요일이 없습니다. 다시 입력해주세요.")

elif st.session_state.step == 4:
    st.header("Step 4: 날짜 범위 입력")
    input_date = st.text_input("데이터셋을 구성할 날짜 범위를 입력해주세요 (예시:20230101,20241231)")
    if st.button("완료", key="step4"):
        if not input_date or input_date.strip() == '전체':
            st.session_state.TARGET_DATES = [START_DATE, END_DATE]
            st.session_state.step = 5
        else:
            input_parts = [i.strip() for i in input_date.split(',')]
            if len(input_parts) == 2:
                user_start_date = pd.to_datetime(input_parts[0])
                user_end_date = pd.to_datetime(input_parts[1])
                if user_start_date > user_end_date:
                    st.warning("시작일이 종료일보다 늦습니다. 다시 입력해주세요.")
                elif user_start_date < START_DATE or user_end_date > END_DATE:
                    st.warning(f"기간은 {START_DATE.strftime('%Y%m%d')}부터 {END_DATE.strftime('%Y%m%d')}까지여야 합니다.")
                else:
                    st.session_state.TARGET_DATES = [user_start_date, user_end_date]
                    st.session_state.step = 5
            else:
                st.warning("형식이 올바르지 않습니다. 'YYYYMMDD,YYYYMMDD'로 입력해주세요.")

elif st.session_state.step == 5:
    st.success("모든 입력이 완료되었습니다!")
    st.write(f"선택한 지역: {st.session_state.TARGET_REGION}")
    st.write(f"선택한 스포츠: {st.session_state.TARGET_SPORTS}")
    st.write(f"선택한 요일: {st.session_state.TARGET_WEEKDAYS}")
    st.write(f"선택한 날짜: {st.session_state.TARGET_DATES}")

    if st.session_state.TARGET_DATES and len(st.session_state.TARGET_DATES) == 2:
        date_range = pd.date_range(start=st.session_state.TARGET_DATES[0],
                                   end=st.session_state.TARGET_DATES[1], freq='D')
        st.write(f"선택된 날짜 범위 ({len(date_range)}일):")
        st.write(date_range)
    else:
        st.warning("날짜 범위가 설정되지 않았습니다. 다시 입력해주세요.")




# 3. 입력한 설정을 기반으로 데이터 셋 구성

# 기본 베이스가되는 데이터프레임설정, 날짜 범위 설정
base_df = pd.DataFrame()
base_df['region'] = pd.NA
base_df['date'] = pd.NA
date_range = pd.date_range(start=TARGET_DATES[0], end=TARGET_DATES[1], freq='D')
base_df = pd.DataFrame(columns=[
    'date', 'region', 'accident_count', 'injury_count', 'death_count', 'game_count',
    'sports_type', 'temperature', 'precipitation', 'snow_depth', 'weather_condition',
    'is_post_season', 'is_hometeam_win', 'is_holiday', 'weekday', 'audience',
    'game_start_time', 'game_end_time'
])
final_df = base_df.copy()

for region in TARGET_REGION:
    temp_df = base_df.copy()
    temp_df = pd.DataFrame({'date': date_range})
    temp_df['region'] = region
    
    # 스타디움 정보 가져오기
    if not stadium_df.empty:
        stadiums_in_target_region = stadium_df[stadium_df['region'] == region]
        stadium_codes_in_region = stadiums_in_target_region['stadium_code'].unique().tolist()
        #st.write(f"\n{TARGET_REGION} 내 경기장 코드: {stadium_codes_in_region}")
    else:
        stadium_codes_in_region = []
        #st.write(f"\n{TARGET_REGION} 내 경기장 정보 없음 또는 stadium 테이블 로드 실패.")
        
    # 스포츠경기 정보 가져오기
    if not sports_game_df.empty and stadium_codes_in_region:
        games_in_region_df = sports_game_df[sports_game_df['stadium_code'].isin(stadium_codes_in_region)]
        games_in_region_df = games_in_region_df.rename(columns={'game_date': 'date'})
        #st.write(games_in_region_df)
        if not games_in_region_df.empty:
            games_in_region_df["match_type"] = (
                games_in_region_df["match_type"]
                    .replace({"페넌트레이스": "정규시즌",
                            "순위결정전": "정규시즌",
                            "순위결정정": "정규시즌",   # 오타까지 함께 처리
                            '조별리그' : "정규시즌",
                            "0": "정규시즌"})
                    # ➋ 라운드 표기(1R ~ 33R 등) → 정규시즌
                    .str.replace(r"^\d+R$", "정규시즌", regex=True)
            )
            games_in_region_df["match_type"] = (
                games_in_region_df["match_type"]
                    .replace({'와일드카드':"포스트시즌",
                            '준플레이오프':"포스트시즌", 
                            '플레이오프':"포스트시즌", 
                            '한국시리즈':"포스트시즌",
                            '파이널 라운드A':"포스트시즌",
                            '파이널 라운드B':"포스트시즌",
                            '챔피언결정전':"포스트시즌", 
                            '준결승':"포스트시즌", 
                            '결승':"포스트시즌"})
            )
            game_summary_df = games_in_region_df.groupby('date').agg(
                game_count=('stadium_code', 'count'),
                sports_types_list=('sports_type', lambda x: list(set(x))),
                is_post_season_list=('match_type', lambda x: 1 if any('포스트시즌' in str(mt).lower() for mt in x) else 0),
                game_start_time_agg=('start_time', 'min'),
                game_end_time_agg=('end_time', 'max'),
                is_hometeam_win_agg=('home_team_win', 'max'),
                audience_agg=('audience', 'sum')
            ).reset_index()
            game_summary_df['sports_type'] = game_summary_df['sports_types_list'].apply(lambda x: ','.join(sorted(x)) if x else '없음')
            game_summary_df['is_post_season'] = game_summary_df['is_post_season_list'].astype(int)
            game_summary_df['game_start_time'] = game_summary_df['game_start_time_agg']
            game_summary_df['game_end_time'] = game_summary_df['game_end_time_agg']
            game_summary_df['is_hometeam_win'] = game_summary_df['is_hometeam_win_agg'].astype(int)
            game_summary_df['audience'] = game_summary_df['audience_agg'].astype(int)
            game_summary_df = game_summary_df[['date', 'game_count', 'sports_type', 'is_post_season', 'game_start_time', 'game_end_time', 'is_hometeam_win', 'audience']]
            temp_df = pd.merge(temp_df, game_summary_df, on='date', how='left')
        else:
            st.write(f"{region} 내 해당 기간 경기 정보 없음.")
    else:
        st.write("sports_game_df 로드 실패 또는 대상 지역 내 경기장 없음.")

    temp_df['game_count'] = temp_df['game_count'].fillna(0).astype(int)
    temp_df['sports_type'] = temp_df['sports_type'].fillna('없음')
    temp_df['is_post_season'] = temp_df['is_post_season'].fillna(0).astype(int)
    temp_df['game_start_time'] = temp_df['game_start_time'].fillna(pd.NA) # 또는 적절한 기본값 (예: '정보없음')
    temp_df['game_end_time'] = temp_df['game_end_time'].fillna(pd.NA)   # 또는 적절한 기본값
    temp_df['is_hometeam_win'] = temp_df['is_hometeam_win'].fillna(0).astype(int) # 경기가 없으면 홈팀 승리도 0
    temp_df['audience'] = temp_df['audience'].fillna(0).astype(int)
    
    # 교통사고 데이터 가져오기
    if not traffic_accident_df.empty:
        accidents_in_region_df = traffic_accident_df[traffic_accident_df['region'] == region]
        accidents_in_region_df = accidents_in_region_df.rename(columns={'accident_date': 'date'})
        if not accidents_in_region_df.empty:
            accident_summary_df = accidents_in_region_df.groupby('date').agg(
                accident_count_sum=('accident_count', 'sum'),
                injury_count_sum=('injury_count', 'sum'),
                death_count_sum=('death_count', 'sum')
            ).reset_index()
            accident_summary_df = accident_summary_df.rename(columns={'accident_count_sum': 'accident_count', 'injury_count_sum': 'injury_count', 'death_count_sum': 'death_count'})
            
            temp_df = pd.merge(temp_df, accident_summary_df, on='date', how='left')
        else:
            st.write(f"{region} 내 해당 기간 교통사고 정보 없음.")
            temp_df['accident_count'] = 0
            temp_df['injury_count'] = 0
            temp_df['death_count'] = 0
    else:
        st.write("traffic_accident_df 로드 실패.")
        temp_df['accident_count'] = 0
        temp_df['injury_count'] = 0
        temp_df['death_count'] = 0
        
    temp_df['accident_count'] = temp_df['accident_count'].fillna(0).astype(int)
    temp_df['injury_count'] = temp_df['injury_count'].fillna(0).astype(int)
    temp_df['death_count'] = temp_df['death_count'].fillna(0).astype(int)
    
    # 날씨 데이터 가져오기
    if not weather_df.empty:
        mask = weather_df['region'].apply(lambda x: x in region)
        weather_region_df = weather_df[mask]
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
                    return '비'
                elif pd.notna(row['avg_cloud_amount']):
                    if row['avg_cloud_amount'] >= 7: # (0-10 기준)
                        return '흐림'
                    elif row['avg_cloud_amount'] >= 3:
                        return '약간흐림' # 또는 '약간흐림' 등
                    else:
                        return '맑음'
                return '정보없음' # 강수량 없고 구름 정보도 없는 경우
                
            weather_summary_df['weather_condition'] = weather_summary_df.apply(get_weather_condition, axis=1)
            weather_summary_df = weather_summary_df[['date', 'temperature', 'precipitation', 'snow_depth', 'weather_condition']]
            temp_df = pd.merge(temp_df, weather_summary_df, on='date', how='left')
        else:
            st.write(f"{region} 내 해당 기간 날씨 정보 없음.")
            temp_df['temperature'] = np.nan
            temp_df['precipitation'] = np.nan
            temp_df['snow_depth'] = np.nan
            temp_df['weather_condition'] = '정보없음'
    else:
        st.write("weather_df 로드 실패.")
        temp_df['temperature'] = np.nan
        temp_df['precipitation'] = np.nan
        temp_df['snow_depth'] = np.nan
        temp_df['weather_condition'] = '정보없음'
    
    # 공휴일 및 주말 데이터 가져오기
    weekday_map_kr = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
    temp_df['weekday'] = temp_df['date'].dt.dayofweek.map(weekday_map_kr)
    try:
        import holidays
        # temp_df['date']에서 연도를 뽑아와서 unique 값으로 추출
        kr_holidays = holidays.KR(years=temp_df['date'].dt.year.unique().tolist()) 
        is_statutory_holiday = temp_df['date'].apply(lambda d: d in kr_holidays)
        is_saturday = (temp_df['weekday'] == '토')
        is_sunday = (temp_df['weekday'] == '일')
        # 3. 세 가지 조건을 OR 연산자로 결합하여 is_holiday 컬럼 생성
        # (하나라도 True이면 True -> 1, 모두 False이면 False -> 0)
        temp_df['is_holiday'] = (is_statutory_holiday | is_saturday | is_sunday).astype(int)
    except ImportError:
        st.write("`holidays` 라이브러리가 설치되지 않았습니다. `pip install holidays`로 설치해주세요. 'is_holiday'는 0으로 처리됩니다.")
        temp_df['is_holiday'] = 0
    except Exception as e:
        st.write(f"공휴일 정보 처리 중 오류: {e}. 'is_holiday'는 0으로 처리됩니다.")
        temp_df['is_holiday'] = 0

    final_df = pd.merge(final_df,temp_df,how='outer')

final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
final_df['temperature'] = pd.to_numeric(final_df['temperature'], errors='coerce').round(1)
final_df['precipitation'] = pd.to_numeric(final_df['precipitation'], errors='coerce').round(1)
final_df['snow_depth'] = pd.to_numeric(final_df['precipitation'], errors='coerce').round(1)

# 4. 최종 데이터 셋 csv 파일로 저장 및 DataFrame 출력 

from datetime import datetime
fn_region , fn_sports, fn_weekdays = None, None, None
fn_dates = f"{TARGET_DATES[0].strftime('%Y%m%d')}~{TARGET_DATES[1].strftime('%Y%m%d')}"
if len(TARGET_REGION) == len(stadium_df['region'].unique().tolist()):
    fn_region = "전국"
elif len(TARGET_REGION) > 2:
    fn_region = f"{TARGET_REGION[0]}_외_{len(TARGET_REGION)-1}지역"
else:
    fn_region = f"{TARGET_REGION[0]}_{TARGET_REGION[1]}"
if len(TARGET_SPORTS) == len(stadium_df['sports_type'].unique().tolist()):
    fn_sports = "전종목"
elif len(TARGET_SPORTS) > 2:
    fn_sports = f"{TARGET_SPORTS[0]}_외_{len(TARGET_SPORTS)-1}종목"
else:
    fn_sports = f"{TARGET_SPORTS[0]}_{TARGET_SPORTS[1]}"
if len(TARGET_WEEKDAYS) == 7:
    fn_weekdays = "전체요일"
elif len(TARGET_WEEKDAYS) > 2:
    fn_weekdays = f"{TARGET_WEEKDAYS[0]}_외_{len(TARGET_WEEKDAYS)-1}요일"
else:
    fn_weekdays = f"{TARGET_WEEKDAYS[0]}_{TARGET_WEEKDAYS[1]}"
filename = f"{fn_dates}_{fn_region}_{fn_sports}_{fn_weekdays}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
final_df.to_csv(f"./results/2nd-dataset_{filename}", index=False)
st.write(f"{filename} 파일 생성이 완료되었습니다.")
st.write("\n--- Final Dataset ---")
st.dataframe(final_df)



# 기존 코드로 파일 저장
final_df.to_csv(f"./results/2nd-dataset_{filename}", index=False)
st.write(f"✅ {filename} 파일 생성이 완료되었습니다.")
st.write("\n--- Final Dataset ---")
st.dataframe(final_df)

# 다운로드 버튼 추가
csv_buffer = io.StringIO()
final_df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

st.download_button(
    label = f"⬇️ {filename} 다운로드",
    data = csv_data,
    file_name = f"2nd-dataset_{filename}",
    mime = "text/csv"
)

