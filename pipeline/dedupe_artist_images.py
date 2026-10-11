from dotenv import load_dotenv
import os
import numpy as np
import psycopg2

# -----------------------------
# config
# -----------------------------

SIMILARITY_THRESHOLD = 0.995  # 이 이상이면 "같은 그림(재업로드)"으로 판단

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


def parse_vec(v):
    if isinstance(v, str):
        return np.array(v.strip("[]").split(","), dtype=np.float32)
    return np.array(v, dtype=np.float32)


def get_artist_ids():
    cur.execute("SELECT DISTINCT artist_id FROM artwork_embeddings ORDER BY artist_id")
    return [r[0] for r in cur.fetchall()]


def get_artist_vectors(artist_id):
    cur.execute("""
        SELECT img_id, embd_vect
        FROM artwork_embeddings
        WHERE artist_id = %s
        ORDER BY img_id
    """, (artist_id,))
    return [(r[0], parse_vec(r[1])) for r in cur.fetchall()]


def find_duplicates(vectors):
    # vectors: [(img_id, vec), ...] -> keep 첫 등장, 이후 비슷한 건 제거 대상으로 표시
    removed = set()
    pairs = []

    for i in range(len(vectors)):
        id_i, v_i = vectors[i]
        if id_i in removed:
            continue

        for j in range(i + 1, len(vectors)):
            id_j, v_j = vectors[j]
            if id_j in removed:
                continue

            sim = float(np.dot(v_i, v_j) / (np.linalg.norm(v_i) * np.linalg.norm(v_j)))

            if sim >= SIMILARITY_THRESHOLD:
                pairs.append((id_i, id_j, round(sim, 4)))
                removed.add(id_j)

    return pairs, removed


def delete_images(img_ids):
    # artwork_embeddings는 artist_images FK(ON DELETE CASCADE)라 같이 삭제됨
    cur.execute(
        "DELETE FROM artist_images WHERE img_id = ANY(%s)",
        (list(img_ids),)
    )


if __name__ == "__main__":

    artist_ids = get_artist_ids()

    all_removed = set()
    report = []

    for artist_id in artist_ids:

        vectors = get_artist_vectors(artist_id)
        pairs, removed = find_duplicates(vectors)

        if pairs:
            report.append((artist_id, len(vectors), pairs))
            all_removed |= removed

    total_dup_pairs = sum(len(p) for _, _, p in report)

    print(f"🔍 중복 걸린 작가 수: {len(report)} / {len(artist_ids)}")
    print(f"🔍 중복 쌍 수: {total_dup_pairs}")
    print(f"🔍 삭제 대상 이미지 수: {len(all_removed)}")
    print()

    report.sort(key=lambda x: len(x[2]), reverse=True)

    for artist_id, img_cnt, pairs in report:
        print(f"  artist_id={artist_id} (이미지 {img_cnt}장) -> 중복쌍 {len(pairs)}개")
        for keep_id, remove_id, sim in pairs:
            print(f"      keep img_id={keep_id}  vs  remove img_id={remove_id}  (sim={sim})")

    print()
    answer = input(f"정말 {len(all_removed)}장을 삭제할까요? (artist_images + artwork_embeddings에서 같이 빠짐) [y/N] ")

    if answer.strip().lower() == "y":

        delete_images(all_removed)
        conn.commit()
        print(f"✅ {len(all_removed)}장 삭제 완료")

    else:

        print("취소했어요. (조회만 수행, 아무것도 지우지 않음)")

    cur.close()
    conn.close()
