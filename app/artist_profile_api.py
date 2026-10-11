from fastapi import APIRouter
import traceback
import numpy as np
import torch
import os
from dotenv import load_dotenv
import ast
import requests
from io import BytesIO
import boto3

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from app.artist_profile_model import (
    ArtistProfileRequest,
    ArtistProfileResponse
)

from collections import Counter
import psycopg2

from app.clip_loader import (
    model,
    processor,
    device
)

router = APIRouter()


# -----------------------------
# PostgreSQL 연결
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


def load_image_from_ncp_url(img_url):

    key = img_url.split(
        f"{os.getenv('NCP_BUCKET')}/"
    )[1]

    obj = s3.get_object(
        Bucket=os.getenv("NCP_BUCKET"),
        Key=key
    )

    return Image.open(
        BytesIO(obj["Body"].read())
    ).convert("RGB")


# -----------------------------
# palette 추출
# -----------------------------
def extract_palette(
        image_path,
        color_count=5
):

    image = load_image_from_ncp_url(
        image_path
    )

    image = image.resize(
        (150, 150)
    )

    pixels = list(
        image.getdata()
    )

    counter = Counter(
        pixels
    )

    most_common = counter.most_common(
        color_count
    )

    result = []

    for color, _ in most_common:

        result.append({

            "r": color[0],

            "g": color[1],

            "b": color[2]
        })

    return result


# -----------------------------
# artist embedding 조회
# -----------------------------
def get_artist_embedding(
        artist_id
):

    conn = get_conn()

    cur = conn.cursor()

    cur.execute(
        """
        SELECT embd_vect
        FROM artist_embeddings
        WHERE artist_id = %s
        """,
        (artist_id,)
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:

        return None

    vec = row[0]

    if isinstance(vec, str):
        vec = ast.literal_eval(vec)

    return np.array(
        vec,
        dtype=np.float32
    )


# -----------------------------
# artist top tags 조회
# -----------------------------
def get_artist_top_tags(
        artist_id
):

    conn = get_conn()

    cur = conn.cursor()

    cur.execute(
        """
        SELECT t.tag_name
        FROM artist_tags at
        JOIN tags t
            ON at.tag_id = t.tag_id
        WHERE at.artist_id = %s
        ORDER BY at.avg_score DESC
        LIMIT 5
        """,
        (artist_id,)
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        r[0]
        for r in rows
    ]


# -----------------------------
# 이미지 embedding 생성
# -----------------------------
def get_image_embedding(img_url):

    image = load_image_from_ncp_url(
        img_url
    )

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        image_features = model.get_image_features(
            **inputs
        )

    if not isinstance(
        image_features,
        torch.Tensor
    ):
        image_features = (
            image_features.pooler_output
        )

    image_features = (
        image_features
        /
        image_features.norm(
            dim=-1,
            keepdim=True
        )
    )

    return image_features.cpu().numpy()[0]


# -----------------------------
# 작가 프로필 생성 API
# -----------------------------
@router.post(
    "/artist/profile/build",
    response_model=ArtistProfileResponse
)
def build_artist_profile(
        req: ArtistProfileRequest
):

    try:

        print("🎨 ARTIST PROFILE BUILD")
        print(req)

        # =========================
        # 이미지 없는 경우
        # =========================
        if not req.imageUrls:

            return ArtistProfileResponse(

                styleDesc="이미지 없음",

                colors=[],

                tags=[],

                profileImg=""
            )

        # =========================
        # 작가 centroid embedding
        # =========================
        artist_emb = get_artist_embedding(
            req.artistId
        )

        if artist_emb is None:

            return ArtistProfileResponse(

                styleDesc="임베딩 없음",

                colors=[],

                tags=[],

                profileImg=""
            )

        # =========================
        # 대표 이미지 선정
        # =========================
        best_similarity = -1

        profile_img = ""

        for img_url in req.imageUrls:

            print("IMG =", img_url)

            try:

                img_emb = get_image_embedding(
                    img_url
                )

                sim = cosine_similarity(
                    [artist_emb],
                    [img_emb]
                )[0][0]

                print(
                    f"SIM = {sim} / {img_url}"
                )

                if sim > best_similarity:

                    best_similarity = sim

                    profile_img = img_url

            except Exception as e:

                print("❌ IMAGE ERROR =", img_url)
                print(e)

                traceback.print_exc()

                continue

        print(
            "🖼 PROFILE IMG =",
            profile_img
        )

        # =========================
        # 대표 이미지 없는 경우
        # =========================
        if not profile_img:

            return ArtistProfileResponse(

                styleDesc="대표 이미지 선정 실패",

                colors=[],

                tags=[],

                profileImg=""
            )

        # =========================
        # 태그 조회
        # =========================
        tags = get_artist_top_tags(
            req.artistId
        )

        print(
            "🏷 TAGS =",
            tags
        )

        # =========================
        # palette 추출
        # =========================
        palette = extract_palette(
            profile_img
        )

        colors = [

            "#{:02X}{:02X}{:02X}".format(
                c["r"],
                c["g"],
                c["b"]
            )

            for c in palette
        ]

        print(
            "🎨 COLORS =",
            colors
        )

        # =========================
        # 스타일 설명
        # =========================
        tag_text = ", ".join(
            [
                f"#{t}"
                for t in tags
            ]
        )

        style_desc = f"""
{req.artistName} 특유의 감성적인 인물 표현과
세련된 색채 구성이 중심이 되는 스타일입니다.

특히 {tag_text} 기반의 표현이 두드러지며,

부드러운 광원과 감각적인 캐릭터 연출이
조화롭게 어우러지는 분위기를 형성합니다.
""".strip()

        print("✅ PROFILE GENERATED")

        # =========================
        # 반환
        # =========================
        return ArtistProfileResponse(

            styleDesc=style_desc,

            colors=colors,

            tags=tags,

            profileImg=profile_img
        )

    except Exception:

        traceback.print_exc()

        return ArtistProfileResponse(

            styleDesc="생성 실패",

            colors=[],

            tags=[],

            profileImg=""
        )