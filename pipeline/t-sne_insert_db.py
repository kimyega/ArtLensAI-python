import numpy as np
import psycopg2
import ast
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv

# -----------------------------
# DB CONNECT
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

print("🔥 LOAD EMBEDDINGS FROM DB")

# -----------------------------
# 1. DB에서 embedding 가져오기
# -----------------------------
cur.execute("""
    SELECT
        ae.artist_id,
        a.name,
        ae.embd_vect
    FROM artist_embeddings ae
    JOIN artists a
        ON ae.artist_id = a.artist_id
""")

rows = cur.fetchall()

artist_ids = []
artist_names = []
X = []

for artist_id, artist_name, vec in rows:
    # vector가 문자열로 들어오는 경우 처리
    if isinstance(vec, str):
        vec = ast.literal_eval(vec)

    X.append(np.array(vec, dtype=np.float32))
    artist_ids.append(artist_id)
    artist_names.append(artist_name)

X = np.array(X)

print(
    f"artists={len(artist_ids)} "
    f"embedding_shape={X.shape}"
)

print("🔥 running t-SNE...")

# -----------------------------
# 2. t-SNE
# -----------------------------
perplexity = min(10, len(X) - 1)

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42,
    init="pca"
)

X_2d = tsne.fit_transform(X)

# -----------------------------
# 3. clustering (cluster_label용)
# -----------------------------
cluster_count = min(5, len(X))

kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# -----------------------------
# 4. DB INSERT
# -----------------------------
print("🔥 saving to DB...")

for i in range(len(artist_ids)):
    artist_id = artist_ids[i]
    x = float(X_2d[i][0])
    y = float(X_2d[i][1])
    cluster_label = f"cluster_{labels[i]}"

    cur.execute("""
        INSERT INTO artist_coords (artist_id, x, y, cluster_label)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (artist_id)
        DO UPDATE SET
            x = EXCLUDED.x,
            y = EXCLUDED.y,
            cluster_label = EXCLUDED.cluster_label,
            updated_at = NOW()
    """, (artist_id, x, y, cluster_label))

conn.commit()

cur.close()
conn.close()

print("🔥 DONE -> artist_coords saved")