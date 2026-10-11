import os
from dotenv import load_dotenv
import psycopg2

# -----------------------------
# 로컬 이미지 루트
# -----------------------------
SAVE_ROOT = "dataset/images"

print(os.getcwd())
print(os.path.abspath(SAVE_ROOT))

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
# artist_id 조회
# -----------------------------
def get_artist_id(name):

    cur.execute(
        """
        SELECT artist_id
        FROM artists
        WHERE name = %s
        """,
        (name,)
    )

    row = cur.fetchone()

    return row[0] if row else None


# -----------------------------
# DB 저장
# -----------------------------
def insert_image(
        artist_id,
        img_url,
        img_key
):

    cur.execute(
        """
        INSERT INTO artist_images (
            artist_id,
            img_url,
            img_key,
            is_primary
        )
        VALUES (
            %s,
            %s,
            %s,
            %s
        )
        ON CONFLICT DO NOTHING
        """,
        (
            artist_id,
            img_url,
            img_key,
            False
        )
    )


# -----------------------------
# rebuild
# -----------------------------
def rebuild_db():

    artists = os.listdir(
        SAVE_ROOT
    )

    print(
        f"🔥 artists count = {len(artists)}"
    )

    for artist in artists:

        artist_path = os.path.join(
            SAVE_ROOT,
            artist
        )

        if not os.path.isdir(
                artist_path
        ):
            continue

        print(
            f"\n🎨 artist = {artist}"
        )

        artist_id = get_artist_id(
            artist
        )

        if artist_id is None:

            print(
                f"❌ artist not found = {artist}"
            )

            continue

        files = os.listdir(
            artist_path
        )

        success_count = 0

        for file_name in files:

            file_path = os.path.join(
                artist_path,
                file_name
            )

            if not os.path.isfile(
                    file_path
            ):
                continue

            ext = os.path.splitext(
                file_name
            )[1].lower()

            if ext not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            ]:
                continue

            img_key = (
                f"artists/"
                f"{artist}/"
                f"{file_name}"
            )

            insert_image(

                artist_id=artist_id,

                img_url=file_path,

                img_key=img_key
            )

            success_count += 1

        conn.commit()

        print(
            f"✅ inserted = {success_count}"
        )


# -----------------------------
# run
# -----------------------------
if __name__ == "__main__":

    rebuild_db()

    cur.close()
    conn.close()

    print("\n🔥 ALL DONE")