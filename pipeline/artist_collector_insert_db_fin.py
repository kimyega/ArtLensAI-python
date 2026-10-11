import requests
import xml.etree.ElementTree as ET
import time
import psycopg2
import os
from dotenv import load_dotenv

BASE_URL = "https://safebooru.org/index.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# -----------------------------
# 고정 작가들
# -----------------------------

FIXED_ARTISTS = {
    "tokki",
    "mika_pikazo",
    "hiten_(hitenkei)",
    "wlop",
    "anmi",
    "lack",
    "kantoku",
    "fuzichoco",
    "yoneyama_mai",
    "redjuice"
}

TARGET_COUNT = 100

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
# fetch artists
# -----------------------------
def fetch_artists(limit=100, pid=0):

    params = {
        "page": "dapi",
        "s": "tag",
        "q": "index",
        "type": 1,
        "limit": limit,
        "pid": pid
    }

    for retry in range(3):

        try:

            res = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=30
            )

            if res.status_code != 200:
                return []

            root = ET.fromstring(res.text)

            tags = root.findall("tag")

            result = []

            for t in tags:
                if int(t.attrib.get("type", -1)) != 1:
                    continue

                result.append(
                    (
                        t.attrib.get("name"),
                        int(t.attrib.get("count", 0))
                    )
                )

            return result

        except Exception as e:

            print(
                f"retry {retry+1}/3 pid={pid}",
                e
            )

            time.sleep(3)

    print(f"skip pid={pid}")

    return []


# -----------------------------
# INSERT TO DB
# -----------------------------
def insert_artist(name, count):
    cur.execute("""
        INSERT INTO artists (name, source, post_cnt)
        VALUES (%s, %s, %s)
        ON CONFLICT (name) DO NOTHING
    """, (name, "safebooru", count))


# -----------------------------
# build DB
# -----------------------------
def build_artist_db():

    selected = {}

    # 고정 작가 10명 먼저 넣기
    for artist in FIXED_ARTISTS:
        selected[artist] = 999999999

    pid = 0

    print("🔥 artist collecting start")

    while len(selected) < TARGET_COUNT:

        print(f"[+] fetching pid {pid}")

        data = fetch_artists(limit=1000, pid=pid)

        if not data:
            print(f"skip pid={pid}")
            pid += 1
            continue

        for name, count in data:

            if not name:
                continue

            if count >= 120:

                if name not in selected:
                    print(
                        f"✔ found artist: "
                        f"{name} ({count})"
                    )
                    
                selected[name] = count

        pid += 1

        if pid > 100000:
            print("❌ safety break")
            break

        time.sleep(0.2)

        print(
            f"selected={len(selected)}"
        )

    # ------------------
    # 고정 작가 우선
    # ------------------
    final_artists = []

    for artist in FIXED_ARTISTS:
        if artist in selected:
            final_artists.append(
                (artist, selected[artist])
            )

    # ------------------
    # 나머지 인기 작가
    # ------------------
    others = [
        (name, count)
        for name, count in selected.items()
        if name not in FIXED_ARTISTS
    ]

    others.sort(
        key=lambda x: x[1],
        reverse=True
    )

    remain = TARGET_COUNT - len(final_artists)

    final_artists.extend(
        others[:remain]
    )

    # ------------------
    # DB 저장
    # ------------------
    inserted = 0

    for name, count in final_artists:

        try:
            insert_artist(name, count)

            inserted += 1

            print(
                f"✔ inserted {name} "
                f"({count})"
            )

        except Exception as e:
            print("insert error:", e)

    conn.commit()

    print()
    print("==============")
    print("selected:", len(final_artists))
    print("inserted:", inserted)
    print("==============")

    print()
    print("fixed artists included")

    for artist in sorted(FIXED_ARTISTS):
        print(
            artist,
            "O" if artist in [x[0] for x in final_artists]
            else "X"
        )


# -----------------------------
# run
# -----------------------------
if __name__ == "__main__":
    build_artist_db()

    cur.close()
    conn.close()

    print("🔥 DONE -> artists inserted into DB")