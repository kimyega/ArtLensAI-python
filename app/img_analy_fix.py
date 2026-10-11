from fastapi import FastAPI
from pydantic import BaseModel
import torch
from PIL import Image
import os
import psycopg2
from app.clip_loader import (
    model,
    processor,
    device
)
from colorthief import ColorThief
import traceback
from openai import OpenAI
from dotenv import load_dotenv
import json
from pipeline.tags import (
    TAGS,
    CATEGORY_WEIGHT,
    CATEGORY_TOPK
)

from app.artist_profile_api import (
    router as artist_profile_router
)

import boto3
from io import BytesIO
import tempfile


app = FastAPI()

app.include_router(
    artist_profile_router
)

# -----------------------------
# config
# -----------------------------


device = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv(
    os.path.join(
        os.path.dirname(__file__),
        ".env"
    )
)


# -----------------------------
# DB 연결
# -----------------------------

load_dotenv(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".env"
    )
)

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("NCP_ENDPOINT"),
    aws_access_key_id=os.getenv("NCP_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("NCP_SECRET_KEY")
)

def get_conn():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=os.getenv("POSTGRES_PORT")
    )
    return conn


# -----------------------------
# OPENAI
# -----------------------------
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

print("🔥 OPENAI READY")

# -----------------------------
# TAG 임베딩 생성
# -----------------------------
tag_prompts = [t["prompt"] for t in TAGS]

text_inputs = processor(
    text=tag_prompts,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(device)

text_inputs = {
    k: v.to(device)
    for k, v in text_inputs.items()
}

with torch.no_grad():

    text_features = model.get_text_features(
        **text_inputs
    )

    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output

    text_features = (
        text_features
        / text_features.norm(
            dim=-1,
            keepdim=True
        )
    )

print("🔥 TAG embeddings loaded")

# -----------------------------
# Request DTO
# -----------------------------
class AnalysisRequest(BaseModel):
    imgKey: str

# -----------------------------
# Top5 작가 검색
# -----------------------------
def find_top5_artists(img_emb):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            ae.artist_id,
            a.name,
            1 - (ae.embd_vect <=> %s::vector) AS similarity
        FROM artist_embeddings ae
        JOIN artists a
            ON ae.artist_id = a.artist_id
        ORDER BY ae.embd_vect <=> %s::vector
        LIMIT 5
    """, (
        img_emb.tolist(),
        img_emb.tolist()
    ))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "artistId": r[0],
            "artistName": r[1],
            "similarity": round(float(r[2]), 4)
        }
        for r in rows
    ]

# -----------------------------
# 작가 좌표 조회
# -----------------------------
def get_artist_coords(artist_ids):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            artist_id,
            x,
            y,
            cluster_label
        FROM artist_coords
        WHERE artist_id = ANY(%s)
    """, (artist_ids,))

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return {
        row[0]: {
            "x": float(row[1]),
            "y": float(row[2]),
            "clusterLabel": row[3]
        }
        for row in rows
    }

# -----------------------------
# 사용자 맵 좌표 계산
# -----------------------------
def calculate_user_position(top5):

    artist_ids = [
        artist["artistId"]
        for artist in top5
    ]

    coord_map = get_artist_coords(
        artist_ids
    )

    top_artist_id = top5[0]["artistId"]

    cluster_label = coord_map[
        top_artist_id
    ]["clusterLabel"]

    total_x = 0.0
    total_y = 0.0
    total_weight = 0.0

    for artist in top5:

        artist_id = artist["artistId"]
        similarity = artist["similarity"]

        if artist_id not in coord_map:
            continue

        x = coord_map[
            artist_id
        ]["x"]

        y = coord_map[
            artist_id
        ]["y"]

        total_x += (
            x * similarity
        )

        total_y += (
            y * similarity
        )

        total_weight += similarity

    if total_weight == 0:

        return {
            "x": 0,
            "y": 0
        }

    return {
        "x": round(
            total_x / total_weight,
            4
        ),
        "y": round(
            total_y / total_weight,
            4
        ),
        "clusterLabel": cluster_label
    }

# -----------------------------
# 태그 추론
# -----------------------------
def extract_tags(img_emb):

    emb = torch.tensor(
        img_emb
    ).to(device)

    emb = (
        emb
        / emb.norm(
            dim=-1,
            keepdim=True
        )
    )

    sims = emb @ text_features.T
    sims = sims.squeeze(0)

    category_scores = {}

    # -----------------------------
    # score 계산
    # -----------------------------
    for i, sim in enumerate(sims):

        tag = TAGS[i]

        score = max(
            sim.item(),
            0
        )

        score *= CATEGORY_WEIGHT[
            tag["category"]
        ]

        category = tag["category"]

        if category not in category_scores:
            category_scores[category] = []

        category_scores[category].append({
            "name": tag["name"],
            "category": category,
            "score": score
        })

    # -----------------------------
    # category별 topk
    # -----------------------------
    final_tags = []

    for category, items in category_scores.items():

        items.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        topk = CATEGORY_TOPK[
            category
        ]

        selected = items[:topk]

        final_tags.extend(selected)


    # score 순 정렬
    final_tags.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return [
        {
            "tag": t["name"],
            "category": t["category"],
            "score": round(
                t["score"],
                4
            )
        }
        for t in final_tags
    ]

# -----------------------------
# ColorThief
# -----------------------------
def extract_palette(img_path):

    ct = ColorThief(img_path)

    palette = ct.get_palette(
        color_count=5
    )

    return [
        {
            "r": c[0],
            "g": c[1],
            "b": c[2]
        }
        for c in palette
    ]


# -----------------------------
# GPT COMMENT
# -----------------------------
def generate_comment(
        tags,
        palette,
        top5
):

    try:

        prompt = f"""
당신은 전문 일러스트 평론가입니다.

다음 분석 결과를 기반으로
짧고 세련된 미술 평론을 작성해주세요.

분석 태그:
{tags}

색상 팔레트:
{palette}

유사 작가:
{top5}

규칙

1. 제목 1줄 생성
2. 본문은 3~4문장만 작성
3. 반드시 "입니다/습니다" 체로 작성
4. 태그를 #tag 형태로 자연스럽게 포함
5. 최소 3개의 태그 포함
6. 태그를 실제 화풍/연출 기법처럼 설명
7. 유사 작가와 표현 기법을 연결해서 설명
8. RGB 값이나 숫자 표현 금지
9. 색감은 감성적으로 표현
10. "AI 분석", "태그", "데이터" 표현 금지
11. 미술 잡지 리뷰처럼 자연스럽게 작성
12. JSON만 반환
13. 태그는 반드시 아래 형식으로 작성

<span class="tag-badge">#tag</span>

14. 유사 작가 이름은 반드시 아래 형식으로 작성

<span class="artist-badge">작가명</span>

15. comment는 HTML 문자열 형태로 반환

16. 문단 구분은 <p> 태그만 사용하고
<br> 태그는 사용하지 말 것
17. 림라이트 이런 왜래어 너무 남발하지 말것
18. tag-badge를 감싸는 소괄호 사용하지 말것

예시

"섬세한 빛의 렌더링과 역동적인 구도의 조화"

이 작품은 입체감을 강조하는
#cel_shading 기반 표현이 인상적인 일러스트입니다.

극적인 #dramatic_lighting 연출은
fuzichoco 특유의 화려한 색채 감각을 떠올리게 하며,

부드러운 #watercolor_medium 터치와 자연스러운 동세 표현은
ask 스타일의 감성적인 분위기와도 닮아 있습니다.

전체적으로 따뜻한 색조와 섬세한 표현이 조화를 이루는 완성도 높은 작품입니다.

반드시 아래 JSON 형식으로만 응답


{{
  "title": "...",
  "comment": "..."
}}
"""

        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        print(content)

        result = json.loads(content)

        return result

    except Exception as e:

        print("GPT ERROR:", e)

        return "코멘트 생성 실패"

# -----------------------------
# 분석 API
# -----------------------------
@app.post("/analysis")
def analyze(req: AnalysisRequest):

    temp_path = None

    try:

        print("imgKey =", req.imgKey)

        response = s3.get_object(
            Bucket=os.getenv("NCP_BUCKET"),
            Key=req.imgKey
        )

        img_bytes = response["Body"].read()

        # -----------------------------
        # 이미지 로딩
        # -----------------------------
        image = Image.open(
            BytesIO(img_bytes)
        ).convert("RGB")

        # -----------------------------
        # ColorThief
        # -----------------------------
        
        with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
        ) as tmp:

            tmp.write(img_bytes)

            temp_path = tmp.name

        palette = extract_palette(
            temp_path
        )

        # -----------------------------
        # CLIP 입력
        # -----------------------------
        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        # -----------------------------
        # 이미지 임베딩
        # -----------------------------
        with torch.no_grad():

            image_features = (
                model.get_image_features(
                    **inputs
                )
            )

            if hasattr(
                image_features,
                "pooler_output"
            ):
                image_features = (
                    image_features.pooler_output
                )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        img_emb = (
            image_features
            .cpu()
            .numpy()[0]
        )

        # -----------------------------
        # 태그 추론
        # -----------------------------
        tags = extract_tags(img_emb)

        # -----------------------------
        # 작가 Top5
        # -----------------------------
        top5 = find_top5_artists(
            img_emb
        )

        # -----------------------------
        # Style Map 좌표
        # -----------------------------
        style_map = calculate_user_position(
            top5
        )

        print("STYLE MAP =", style_map)

        # -----------------------------
        # GPT Result
        # -----------------------------
        gpt_result = generate_comment(
            tags,
            palette,
            top5
        )

        print(tags)

        # -----------------------------
        # 반환
        # -----------------------------
        return {
            "result": 1,

            "palette": palette,

            "tags": tags,

            "top5": top5,

            "styleMap": style_map,

            "title": gpt_result["title"],

            "comment": gpt_result["comment"]
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "result": 0,
            "msg": str(e)
        }
    
    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)