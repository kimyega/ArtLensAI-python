import psycopg2
from tags import TAGS
import os
from dotenv import load_dotenv

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


print("HOST =", os.getenv("POSTGRES_HOST"))
print("DB =", os.getenv("POSTGRES_DB"))
print("USER =", os.getenv("POSTGRES_USER"))

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    port=os.getenv("POSTGRES_PORT")
)

cur = conn.cursor()

# -----------------------------
# INSERT TAGS
# -----------------------------
for tag in TAGS:

    cur.execute("""
        INSERT INTO tags (
            tag_name,
            prompt,
            category
        )
        VALUES (%s, %s, %s)
        ON CONFLICT (tag_name)
        DO UPDATE SET
            prompt = EXCLUDED.prompt,
            category = EXCLUDED.category
    """, (
        tag["name"],
        tag["prompt"],
        tag["category"]
    ))

conn.commit()

cur.close()
conn.close()

print("🔥 TAG SEED COMPLETE")