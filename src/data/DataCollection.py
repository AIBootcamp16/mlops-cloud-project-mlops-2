import re
from datetime import datetime
from collections import defaultdict
import os
import json

import requests
import pandas as pd
from sqlalchemy import create_engine, Text, String, text
import pymysql

API_KEY = os.getenv("TMDB_API_KEY")#'2b3ec7f5300b34cb013403cf34dc8f1a'

def get_tmdb_genres():
    url_genres = f'https://api.themoviedb.org/3/genre/movie/list?api_key={API_KEY}&language=en-US'

    response = requests.get(url_genres)
    genres = response.json()['genres']

    tmdb_genres_ids = {
        "액션": 28,
        "모험": 12,
        "애니메이션": 16,
        "코미디": 35,
        "범죄": 80,
        "다큐멘터리": 99,
        "드라마": 18,
        "가족": 10751,
        "판타지": 14,
        "역사": 36,
        "공포": 27,
        "음악": 10402,
        "미스터리": 9648,
        "로맨스": 10749,
        "SF": 878,
        "TV 영화": 10770,
        "스릴러": 53,
        "전쟁": 10752,
        "서부": 37
    }

    for idx, g in enumerate(genres, start=0):
        g['name_kr'] = [k for k, v in tmdb_genres_ids.items() if v == g['id']][0]
        g['index'] = idx  # 순서 번호 추가
        print(g['index'],g['id'], g['name'], g['name_kr'])

    return genres

def fetch_movies_by_genre(movie_genres):
    BASE_URL = 'https://api.themoviedb.org/3/discover/movie'
    today = datetime.today().strftime('%Y-%m-%d')  # 오늘 날짜

    MIN_MOVIES_PER_GENRE = 20  # 장르별 최소 수집 편수

    # 장르별 영화 리스트 저장
    genre_movies = defaultdict(list)

    collected_ids = set()  # 이미 수집된 영화 ID 저장

    for genre in movie_genres:
        genre_id = genre['id']
        page = 1
        while len(genre_movies[genre_id]) < MIN_MOVIES_PER_GENRE:
            params = {
                'api_key': API_KEY,
                'primary_release_date.gte' : '2000-01-01',
                'primary_release_date.lte': today,
                'region': 'KR',
                'language': 'ko-KR',
                'with_genres': genre_id,
                #'sort_by': 'vote_average.desc',
                'vote_count.gte': 10,  # 최소 1명 이상 투표한 영화만
                'page': page
            }
            response = requests.get(BASE_URL, params=params)
            data = response.json()

            # 영화가 없으면 종료
            if 'results' not in data or not data['results']:
                break

            for m in data['results']:
                # vote_average > 0 확인
                title = m.get('title', '')
                vote = m.get('vote_average', 0)
                lang = m.get('original_language', '')
                movie_id = m['id']
                if movie_id in collected_ids:
                    continue  # 이미 수집된 영화면 건너뜀
                #if   (lang == 'ko' or lang == 'en') and (vote > 0):
                if   re.match(r'^[가-힣A-Za-z0-9\s:,\.\-]+$', title) and (vote > 0):
                    genre_movies[genre_id].append({
                        'id': m['id'],
                        'title': m['title'],
                        'genre_ids': m['genre_ids'],
                        'release_date': m.get('release_date'),
                        'vote_average': m.get('vote_average'),
                        'popularity': m.get('popularity'),
                        'vote_count': m.get('vote_count')
                    })
                    collected_ids.add(movie_id)  # 수집 완료 표시
                # 최소 개수 충족하면 break
                if len(genre_movies[genre_id]) >= MIN_MOVIES_PER_GENRE:
                    break

            page += 1
    all_movies = []
    #장르별 수집 영화 데이터 통합
    for genre_id, movies in genre_movies.items():
        all_movies.extend(movies)
    return all_movies

def fetch_movies_by_poster(movies):
    BASE_URL = 'https://api.themoviedb.org/3/movie/'
    # TMDB 이미지 기본 URL
    IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/w500'  # w500: 500px 너비 이미지
    for m in movies:
        movie_id = m['id']
        url = f"{BASE_URL}{movie_id}"
        params = {
            'api_key': API_KEY,
            'language': 'ko-KR'
        }
        response = requests.get(url, params=params)
        data = response.json()
        poster_path = data.get('poster_path')

        if poster_path:
            m['poster_url'] = f'https://image.tmdb.org/t/p/w500{poster_path}'

def insert_movie_to_db(addr, id, pwd, db_name,table_name, movie:pd.DataFrame):
    # MySQL 연결 문자열 생성
    engine = create_engine("mysql+pymysql://{id}:{pwd}@{addr}:3306/{db_name}?charset=utf8mb4")
    # Insert 를 위해 ids 문자열 json 형식으로 변환
    df_movies['genre_ids'] = df_movies['genre_ids'].apply(json.dumps)
    # DataFrame을 SQL 테이블에 삽입
    movie.to_sql('test_movie', con = engine, if_exists="replace", index=False)

if __name__ == "__main__":
    genres = get_tmdb_genres()
    movies = fetch_movies_by_genre(genres)
    fetch_movies_by_poster(movies)

    df_movies = pd.DataFrame(movies)
    df_movies.to_csv('movies.csv', index=False, encoding='utf-8-sig')
    
    SQL_ADDR=os.getenv("SQL_ADDR")
    SQL_ID=os.getenv("SQL_ID")
    SQL_PWD=os.getenv("SQL_PWD")
    SQL_DBNAME=os.getenv("SQL_DB")

    insert_movie_to_db(SQL_ADDR, SQL_ID, SQL_PWD, SQL_DBNAME,'movie_collection', df_movies)

