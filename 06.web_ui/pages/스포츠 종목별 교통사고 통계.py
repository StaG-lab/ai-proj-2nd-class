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

st.image("./06.web_ui/images/soccer-1331843_2.jpg", use_container_width=True)

st.title("스포츠 종목별 교통사고 통계")

# Set Korean font for matplotlib
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # To display minus sign correctly

df = pd.read_csv('./05.modeling/2nd-dataset_전국_20250605_022817.csv')

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
df['is_post_season'] = df['is_post_season'].astype(bool)
df['is_holiday'] = df['is_holiday'].astype(bool)

# Create a column 'has_game' for easier filtering
df['has_game'] = df['game_count'] > 0

# Check consistency: region_code
st.write("\nUnique values in 'region_code':", df['region_code'].unique())
# If 'region_code' is constant, we might drop it for modeling, but keep for EDA context.

# Clean sports_types: "없음" means no sport, otherwise it's a list
# For analysis, we can create dummy variables for common sports
all_sports = set()
for sports_list in df['sports_types'].unique():
    if sports_list != "없음":
        for sport in sports_list.split(','):
            all_sports.add(sport.strip()) # .strip() to remove potential leading/trailing spaces

st.write("\nIdentified unique sport types:", sorted(list(all_sports)))

# Create boolean columns for each identified sport
for sport in sorted(list(all_sports)):
    df[f'is_{sport}'] = df['sports_types'].apply(lambda x: sport in x)

st.write("\nDataFrame with new sport boolean columns (first 5 rows):")
st.write(df.head())

st.markdown("## 3. 피처 엔지니어링 <span style='font-size:16px; color:gray'>_(Feature Engineering (Time-based features))_</span>", unsafe_allow_html=True)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week_num'] = df['date'].dt.dayofweek # Monday=0, Sunday=6

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
st.write(f"Accident count statistics:\n{df['accident_count'].describe()}")

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
    st.write(f"{col} statistics:\n{df[col].describe()}")

# 4.3 Counts of Categorical Features
categorical_cols = ['is_hometeam_win', 'game_count', 'is_post_season', 'weather_condition', 'is_holiday', 'weekday', 'has_game']
for col in categorical_cols:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(y=df[col], order = df[col].value_counts().index) # y for better label readability
    ax.set_title(f'Counts of {col}')
    ax.set_xlabel('Count')
    ax.set_ylabel(col)
    ax.grid(axis='x')
    st.pyplot(fig)
    st.write(f"\nValue counts for {col}:\n{df[col].value_counts(normalize=True)*100} %")

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
sns.boxplot(x='weekday', y='accident_count', data=df, order=weekday_order)
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
sns.boxplot(x='is_post_season', y='accident_count', data=df[df['has_game']]) # Only for days with games
ax.set_title('Accident Counts: Playoff Game vs. Regular Game (on game days)')
ax.set_xlabel('Is it a Playoff Game?')
ax.set_ylabel('Accident Count')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Regular Game', 'Playoff Game'])
ax.grid(True)
st.pyplot(fig)
st.write(df[df['has_game']].groupby('is_post_season')['accident_count'].describe())

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
numerical_cols_for_corr = ['accident_count', 'audience', 'injury_count','death_count', 'game_count', 'temperature', 'precipitation', 'snow_depth', 'month', 'day_of_week_num']
# Add boolean columns (0/1) for correlation
numerical_cols_for_corr.extend(['is_post_season', 'is_holiday', 'has_game'])
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
    sns.boxplot(x='weather_condition', y='accident_count', hue='is_post_season', 
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
st.write(high_accident_days[['date', 'accident_count', 'game_count', 'sports_types', 'is_post_season', 'temperature', 'precipitation', 'snow_depth', 'weather_condition', 'is_holiday', 'weekday']].sort_values(by='accident_count', ascending=False))

st.write("\nWeather conditions on high accident days:")
st.write(high_accident_days['weather_condition'].value_counts(normalize=True)*100)

st.write("\nGame presence on high accident days:")
st.write(high_accident_days['has_game'].value_counts(normalize=True)*100)

st.write("\nPlayoff presence on high accident (game) days:")
if high_accident_days['has_game'].any():
    st.write(high_accident_days[high_accident_days['has_game']]['is_post_season'].value_counts(normalize=True)*100)