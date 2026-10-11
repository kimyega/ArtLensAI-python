import requests
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv
import time
import psycopg2
import boto3
from urllib.parse import urlparse

BASE_URL = "https://safebooru.org/index.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

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

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    port=os.getenv("POSTGRES_PORT")
)

cur = conn.cursor()

# -----------------------------
# S3 Client 생성
# -----------------------------

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("NCP_ENDPOINT"),
    aws_access_key_id=os.getenv("NCP_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("NCP_SECRET_KEY")
)

# -----------------------------
# artist_id 조회
# -----------------------------
def get_artist_id(name):
    cur.execute("""
        SELECT artist_id
        FROM artists
        WHERE name = %s
    """, (name,))

    row = cur.fetchone()
    return row[0] if row else None


def has_images(artist_id):
    cur.execute("""
        SELECT EXISTS(
            SELECT 1
            FROM artist_images
            WHERE artist_id = %s
        )
    """, (artist_id,))

    return cur.fetchone()[0]

# -----------------------------
# Safebooru 이미지 fetch
# -----------------------------
def fetch_images(artist, limit=100):
    params = {
        "page": "dapi",
        "s": "post",
        "q": "index",
        "tags": artist,
        "limit": limit
    }

    try:
        res = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10)

        if res.status_code != 200 or not res.text.strip():
            return []

        root = ET.fromstring(res.text)
        posts = root.findall("post")

        result = []

        for p in posts:
            url = p.attrib.get("file_url")
            post_id = p.attrib.get("id")  # ⭐ 핵심

            if url:
                if url.startswith("//"):
                    url = "https:" + url

                result.append((post_id, url))

        return result

    except Exception as e:
        print("fetch error:", e)
        return []


# -----------------------------
# 이미지 저장
# -----------------------------
def upload_image_from_url(url, key, retry=2):

    bucket = os.getenv("NCP_BUCKET")

    for _ in range(retry):
        try:
            r = requests.get(url, timeout=15)

            if r.status_code != 200:
                continue

            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=r.content
            )

            return (
                f"{os.getenv('NCP_ENDPOINT')}/"
                f"{bucket}/{key}"
            )

        except Exception as e:
            print("upload error:", e)
            time.sleep(0.5)

    return None


# -----------------------------
# DB insert
# -----------------------------
def insert_image(artist_id, img_url, img_key, safebooru_id):
    cur.execute("""
        INSERT INTO artist_images (
            artist_id,
            img_url,
            img_key,
            is_primary,
            safebooru_id
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (
        artist_id,
        img_url,
        img_key,
        False,
        safebooru_id
    ))


# -----------------------------
# artist 이미지 수집
# -----------------------------
def collect_artist_images(artists, limit_per_artist=30):

    for artist in artists:
        print(f"\n🔥 collecting: {artist}")

        artist_id = get_artist_id(artist)

        if not artist_id:
            print("❌ artist not found")
            continue

        if has_images(artist_id):
            print(f"⏩ skip: {artist}")
            continue

        images = fetch_images(artist, limit_per_artist)

        

        for i, (post_id, url) in enumerate(images):

            # -----------------------------
            # S3 저장
            # -----------------------------
            path = urlparse(url).path
            ext = os.path.splitext(path)[1].lower()

            if not ext:
                ext = ".jpg"

            safe_artist = artist.replace("/", "_")
            img_key = f"artists/{safe_artist}/{i:04d}{ext}"

            img_url = upload_image_from_url(
                url,
                img_key
            )

            if not img_url:
                print("✖ upload failed:", url)
                continue

            # -----------------------------
            # DB insert
            # -----------------------------
            insert_image(
                artist_id=artist_id,
                img_url=img_url,
                img_key=img_key,
                safebooru_id=int(post_id) if post_id else None
            )

            print(f"✔ uploaded + inserted: {img_key}")

            time.sleep(0.2)

        conn.commit()


# -----------------------------
# artist load
# -----------------------------
def load_artists():

    cur.execute("""
        SELECT name
        FROM artists
        ORDER BY post_cnt DESC
    """)

    rows = cur.fetchall()

    return [row[0] for row in rows]


# -----------------------------
# run
# -----------------------------
if __name__ == "__main__":

    artists = load_artists()

    collect_artist_images(
        artists,
        limit_per_artist=100
    )

    cur.close()
    conn.close()

    print("🔥 DONE")