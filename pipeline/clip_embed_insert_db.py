from dotenv import load_dotenv
import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import psycopg2
from collections import defaultdict
from tags import TAGS, CATEGORY_WEIGHT
import boto3
from io import BytesIO

# -----------------------------
# config
# -----------------------------

MAX_IMAGES_PER_ARTIST = 100
BATCH_SIZE = 16


device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔥 CLIP → DB PIPELINE START")

# -----------------------------
# DB
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

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    port=os.getenv("POSTGRES_PORT")
)
cur = conn.cursor()

# -----------------------------
# model
# -----------------------------
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()

# -----------------------------
# TEXT embedding (FIXED)
# -----------------------------
tag_prompts = [t["prompt"] for t in TAGS]

text_inputs = processor(
    text=tag_prompts,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(device)

with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)

    # 안전 보정
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# -----------------------------
# DB helpers
# -----------------------------
def get_artist_id(name):
    cur.execute("SELECT artist_id FROM artists WHERE name=%s", (name,))
    row = cur.fetchone()
    return row[0] if row else None


def get_tag_id(tag_name):
    cur.execute("SELECT tag_id FROM tags WHERE tag_name=%s", (tag_name,))
    row = cur.fetchone()
    return row[0] if row else None

def load_artists_from_ncp():

    bucket = os.getenv("NCP_BUCKET")

    result = s3.list_objects_v2(
        Bucket=bucket,
        Prefix="artists/",
        Delimiter="/"
    )

    artists = []

    for p in result.get("CommonPrefixes", []):
        artist = p["Prefix"].replace("artists/", "").rstrip("/")
        artists.append(artist)

    return artists


def get_artist_images(artist_id):

    cur.execute("""
        SELECT img_id, img_key
        FROM artist_images
        WHERE artist_id = %s
        ORDER BY img_id
        LIMIT %s
    """, (artist_id, MAX_IMAGES_PER_ARTIST))

    return cur.fetchall()


def insert_embedding(artist_id, vec, img_cnt):
    cur.execute("""
        INSERT INTO artist_embeddings (artist_id, embd_vect, img_cnt)
        VALUES (%s, %s, %s)
        ON CONFLICT (artist_id)
        DO UPDATE SET
            embd_vect = EXCLUDED.embd_vect,
            img_cnt = EXCLUDED.img_cnt,
            updated_at = NOW()
    """, (artist_id, vec.tolist(), img_cnt))


def insert_artwork_embedding(img_id, artist_id, vec):
    cur.execute("""
        INSERT INTO artwork_embeddings (img_id, artist_id, embd_vect)
        VALUES (%s, %s, %s)
        ON CONFLICT (img_id)
        DO UPDATE SET
            embd_vect = EXCLUDED.embd_vect,
            updated_at = NOW()
    """, (img_id, artist_id, vec.tolist()))


def insert_tag(artist_id, tag_id, cnt, avg_score):

    cur.execute("""
        INSERT INTO artist_tags (
            artist_id,
            tag_id,
            tag_cnt,
            avg_score
        )
        VALUES (%s, %s, %s, %s)

        ON CONFLICT (artist_id, tag_id)
        DO UPDATE SET
            tag_cnt = EXCLUDED.tag_cnt,
            avg_score = EXCLUDED.avg_score
    """, (
        artist_id,
        tag_id,
        cnt,
        avg_score
    ))


def load_image_from_ncp(key):

    bucket = os.getenv("NCP_BUCKET")

    try:
        obj = s3.get_object(
            Bucket=bucket,
            Key=key
        )

        data = obj["Body"].read()

        return Image.open(
            BytesIO(data)
        ).convert("RGB")

    except Exception as e:
        print("❌ IMAGE LOAD ERROR:", key)
        print(e)
        raise

# -----------------------------
# batch image embedding (FIXED)
# -----------------------------
def get_batch_embedding(items):
    # items: list of (img_id, img_key)
    ok_ids = []
    images = []

    for img_id, key in items:
        try:
            images.append(load_image_from_ncp(key))
            ok_ids.append(img_id)
        except Exception as e:
            print("skip:", key)
            continue

    if not images:
        return [], np.empty((0, 512))

    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        feats = model.get_image_features(**inputs)

    # 🔥 핵심: 무조건 tensor 보장
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output

    feats = feats / feats.norm(dim=-1, keepdim=True)

    return ok_ids, feats.cpu().numpy()


# -----------------------------
# tag inference
# -----------------------------
def get_tags(img_emb, topk=5):

    img_emb = torch.tensor(img_emb).to(device)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    sims = img_emb @ text_features.T
    sims = sims.squeeze(0)

    all_scores = []

    # -----------------------------
    # score 계산
    # -----------------------------
    for i, sim in enumerate(sims):

        tag = TAGS[i]

        score = sim.item()

        # 음수 방지
        score = max(score, 0)

        # category weight 적용
        score *= CATEGORY_WEIGHT[tag["category"]]

        all_scores.append({
            "tag": tag,
            "score": score
        })

    # -----------------------------
    # 전체 정렬
    # -----------------------------
    all_scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    top_tags = all_scores[:topk]

    return top_tags


# -----------------------------
# artist pipeline
# -----------------------------
def process_artist(artist):

    artist_id = get_artist_id(artist)
    if artist_id is None:
        print("❌ artist not in DB:", artist)
        return

    rows = get_artist_images(artist_id)

    print(f"📸 image count : {len(rows)}")

    for img_id, img_key in rows[:5]:
        print("   ", os.path.basename(img_key))

    if not rows:
        return

    embeddings = []

    tag_stats = defaultdict(lambda: {
        "count": 0,
        "score_sum": 0.0
    })

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i+BATCH_SIZE]

        ok_ids, emb_batch = get_batch_embedding(batch)

        if len(emb_batch) == 0:
            continue

        embeddings.append(emb_batch)

        for img_id, emb in zip(ok_ids, emb_batch):

            insert_artwork_embedding(img_id, artist_id, emb)

            tags = get_tags(emb, topk=5)

            for t in tags:

                tag_name = t["tag"]["name"]
                score = t["score"]

                tag_stats[tag_name]["count"] += 1
                tag_stats[tag_name]["score_sum"] += score


    if not embeddings:
        print("❌ no valid images:", artist)
        return

    embeddings = np.vstack(embeddings)
    avg_emb = np.mean(embeddings, axis=0).astype(np.float32)

    insert_embedding(artist_id, avg_emb, len(rows))

    for tag, stat in tag_stats.items():

        tag_id = get_tag_id(tag)

        if tag_id:

            cnt = stat["count"]

            avg_score = (
                stat["score_sum"] / cnt
            )

            insert_tag(
                artist_id,
                tag_id,
                cnt,
                avg_score
            )

    print("✔ done:", artist)


# -----------------------------
# main
# -----------------------------
if __name__ == "__main__":

    artists = load_artists_from_ncp()

    print("🔥 artists:", len(artists))

    for a in artists:
        print("\nprocessing:", a)
        process_artist(a)

    conn.commit()
    cur.close()
    conn.close()

    print("🔥 DONE → DB saved")