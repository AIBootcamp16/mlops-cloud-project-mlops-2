# 사용자 선호도 기반 음악 추천 시스템

<br>

## 💻 프로젝트 소개
### 🎧 Seed-based Music Recommendation System

사용자가 **검색으로 노래 5곡을 선택**하면, 해당 곡들의 **오디오 특성(Audio Features)**을 기반으로  
콘텐츠 기반 추천(Content-based Filtering)을 수행하여 **맞춤형 추천 리스트**를 제공합니다.  

Spotify API를 활용하여 **검색/메타데이터/오디오 특성**을 가져오며,  
추천 알고리즘은 **코사인 유사도(Cosine Similarity)**를 기반으로 합니다.  

<br>

## ✨ Features
- 🔍 **검색(Search)**: Spotify API를 통한 트랙 검색
- 🎶 **Seed Selection**: 사용자가 좋아하는 노래 5곡 선택
- 🧩 **프로필 벡터 생성**: 선택한 곡의 오디오 특성을 평균화하여 사용자 프로필 구성
- 📊 **추천 리스트 생성**: 코사인 유사도로 후보 카탈로그와 비교하여 Top-K 추천
- 🎧 **UI 제공**: Streamlit으로 간단한 웹 UI
- ⚡ **API 제공**: FastAPI 기반 추천 API

<br>

## 👨‍👩‍👦‍👦 팀 구성원

| ![김소은](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김재록](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김종화](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최보경](https://avatars.githubusercontent.com/u/156163982?v=4) | ![황은혜](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김소은](https://github.com/oriori88)             |            [김재록](https://github.com/UpstageAILab)             |            [김종화](https://github.com/UpstageAILab)             |            [최보경](https://github.com/UpstageAILab)             |            [황은혜](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

<br>

## 🔨 개발 환경 및 기술 스택
- 주 언어 : python
- 버전 및 이슈관리 : github
- 협업 툴 : github, Jira

<br>

## 📁 프로젝트 구조
```
mlops-cloud-project-mlops-2/
  ├─ dataset/
  │   ├─ raw/                  # Spotify 원본 JSON
  │   └─ processed/            # 전처리된 parquet (audio_features, catalog 등)
  ├─ docker/
  ├─ docs/
  ├─ notebooks/
  ├─ src/
  │   ├─ api
  │   ├─ data
  │   ├─ model
  │   └─ utils
  ├─ web/
  ├─ .env.template
  ├─ requirements.txt
  └─ README.md
```

<br>

## 💻​ 구현 기능
### 기능1
- _작품에 대한 주요 기능을 작성해주세요_
### 기능2
- _작품에 대한 주요 기능을 작성해주세요_
### 기능3
- _작품에 대한 주요 기능을 작성해주세요_

<br>

## 🛠️ 작품 아키텍처(필수X)
- #### _아래 이미지는 예시입니다_
![이미지 설명](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*ub_u88a4MB5Uj-9Eb60VNA.jpeg)

<br>

## 🚨​ 트러블 슈팅
### 1. OOO 에러 발견

#### 설명
- _프로젝트 진행 중 발생한 트러블에 대해 작성해주세요_

#### 해결
- _프로젝트 진행 중 발생한 트러블 해결방법 대해 작성해주세요_

<br>

## 📌 프로젝트 회고
### 박패캠
- _프로젝트 회고를 작성해주세요_

<br>

## 📰​ 참고자료
- _참고자료를 첨부해주세요_
