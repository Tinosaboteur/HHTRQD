from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pymysql
from datetime import datetime

app = Flask(__name__)

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="01020304",  
        database="ahp_danhnganh",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor
    )

# ---------- Lấy tiêu chí và phương án ----------
def get_criteria_and_alternatives():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, ten_tieu_chi FROM tieu_chi")
    criteria = cursor.fetchall()
    cursor.execute("SELECT id, ten_phuong_an FROM phuong_an")
    alternatives = cursor.fetchall()
    conn.close()
    return criteria, alternatives

# ---------- Lấy ma trận từ form ----------
def compute_pairwise_matrix(prefix, n, form):
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            key = f"{prefix}_{i}_{j}"
            val = float(form.get(key, 1))
            matrix[i][j] = val
            matrix[j][i] = 1 / val
    return matrix

# ---------- Tính trọng số và CR ----------
def ahp_weighting(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = np.argmax(np.real(eigvals))
    max_eigval = np.real(eigvals[max_index])
    weights = np.real(eigvecs[:, max_index])
    weights = weights / np.sum(weights)

    n = matrix.shape[0]
    CI = (max_eigval - n) / (n - 1) if n > 1 else 0
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
               7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0

    return weights, round(CR, 4)

# ---------- Trang nhập ----------
@app.route("/", methods=["GET", "POST"])
def index():
    criteria, alternatives = get_criteria_and_alternatives()
    alt_names = [alt[1] for alt in alternatives]
    alt_ids = [alt[0] for alt in alternatives]

    if request.method == "POST":
        n_crit = len(criteria)
        m_alt = len(alternatives)

        crit_matrix = compute_pairwise_matrix("pc", n_crit, request.form)
        crit_weights, crit_consistency = ahp_weighting(crit_matrix)
        crit_weights = crit_weights.tolist()

        alt_weights_all = []
        alt_consistency_all = []
        for i in range(n_crit):
            alt_matrix = compute_pairwise_matrix(f"alt_pc_{i}", m_alt, request.form)
            weights, consis = ahp_weighting(alt_matrix)
            alt_weights_all.append(weights.tolist())
            alt_consistency_all.append(consis)

        global_scores = {alt_names[i]: 0 for i in range(m_alt)}
        for i in range(n_crit):
            for j in range(m_alt):
                global_scores[alt_names[j]] += crit_weights[i] * alt_weights_all[i][j]

        best_alternative = max(global_scores, key=global_scores.get)

        # Ghi vào cơ sở dữ liệu
        conn = get_connection()
        cursor = conn.cursor()
        for i in range(m_alt):
            cursor.execute("""
                INSERT INTO ket_qua (id_phuong_an, diem_tong_hop, la_tot_nhat, thoi_gian)
                VALUES (%s, %s, %s, %s)
            """, (alt_ids[i], global_scores[alt_names[i]], alt_names[i] == best_alternative, datetime.now()))
        conn.commit()
        conn.close()

        return redirect(url_for("ketqua"))

    return render_template("index.html", criteria=criteria, alternatives=alt_names, enumerate=enumerate)

# ---------- Trang kết quả ----------
@app.route("/ketqua")
def ketqua():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.ten_phuong_an, k.diem_tong_hop, k.la_tot_nhat, k.thoi_gian
        FROM ket_qua k
        JOIN phuong_an p ON k.id_phuong_an = p.id
        ORDER BY k.thoi_gian DESC
        LIMIT 4
    """)
    ket_qua = cursor.fetchall()
    conn.close()
    return render_template("result.html", results=ket_qua)

# ---------- Chạy app ----------
if __name__ == "__main__":
    app.run(debug=True)
