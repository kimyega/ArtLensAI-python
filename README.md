# ArtLensAI-python

[ArtLensAI](https://github.com/kimyega/ArtLensAI)의 화풍 분석을 담당하는 FastAPI 서버입니다.
CLIP 임베딩으로 이미지의 화풍 특징을 추출해 가장 비슷한 작가를 찾아줍니다.

## 하는 일

- 업로드된 이미지를 CLIP으로 임베딩해 색상 팔레트·화풍 태그·유사 작가 분석
- 작가 이미지 수집, 중복 제거, 임베딩 생성, 태그/좌표(t-SNE) DB 반영 등 데이터 파이프라인

## 기술 스택

Python, FastAPI, PyTorch, CLIP, scikit-learn, UMAP, PostgreSQL

## 구조

- `app/` — 화풍 분석 API 서버
- `pipeline/` — 작가 데이터 수집 및 임베딩 파이프라인 스크립트
