from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)

# Danh sách tiêu chí và phương án
criteria = [
    "Tỷ lệ sinh viên có việc làm sau tốt nghiệp",
    "Chi phí học tập",
    "Mức lương khởi điểm",
    "Nhu cầu thị trường",
    "Sở thích cá nhân",
    "Năng lực bản thân"
]

alternatives = [
    "Ngành Khoa học máy tính (Tăng cường TA)",
    "Ngành Công nghệ kỹ thuật hóa học (Tăng cường TA)",
    "Ngành Công nghệ sinh học (Tăng cường TA)",
    "Ngành Vật lý học (Tăng cường TA)"
]

def compute_pairwise_matrix(prefix, n, form):
    """
    Tạo ma trận so sánh cặp kích thước n x n từ dữ liệu nhập.
    Dữ liệu được lấy từ các ô có tên: prefix_i_j với i<j.
    """
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i < j:
                key = f"{prefix}_{i}_{j}"
                try:
                    val = float(form.get(key, 1))
                except:
                    val = 1
                A[i, j] = val
                A[j, i] = 1 / val if val != 0 else 0
    return A

def ahp_weighting(A):
    """
    Tính trọng số theo AHP: chuẩn hóa theo cột, lấy trung bình các hàng.
    Đồng thời tính chỉ số nhất quán.
    """
    n = A.shape[0]
    col_sum = A.sum(axis=0)
    norm_matrix = A / col_sum[np.newaxis, :]
    weights = norm_matrix.mean(axis=1)
    
    # Tính kiểm tra tính nhất quán
    Aw = np.dot(A, weights)
    consistency_vector = Aw / weights
    lambda_max = consistency_vector.mean()
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI_dict = {1:0.0, 2:0.0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0

    return weights, {"lambda_max": lambda_max, "CI": CI, "RI": RI, "cr": CR}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        n_crit = len(criteria)
        m_alt = len(alternatives)
        
        # Phần I: Xử lý ma trận so sánh cặp tiêu chí
        crit_matrix = compute_pairwise_matrix("pc", n_crit, request.form)
        crit_weights, crit_consistency = ahp_weighting(crit_matrix)
        crit_weights = crit_weights.tolist()
        
        # Phần II: Xử lý ma trận so sánh cặp các phương án cho mỗi tiêu chí
        alt_weights_all = []    # Danh sách trọng số các phương án theo từng tiêu chí
        alt_consistency_all = []  # Danh sách kiểm tra nhất quán cho từng tiêu chí
        for i in range(n_crit):
            alt_matrix = compute_pairwise_matrix(f"alt_pc_{i}", m_alt, request.form)
            weights, consis = ahp_weighting(alt_matrix)
            alt_weights_all.append(weights.tolist())
            alt_consistency_all.append(consis)
        
        # Tính điểm tổng hợp cho mỗi phương án:
        # global_score[alt] = sum_{i=0}^{n_crit-1} (crit_weight[i] * alt_weight[i][alt])
        global_scores = {alt: 0 for alt in alternatives}
        for i in range(n_crit):
            for j in range(m_alt):
                global_scores[alternatives[j]] += crit_weights[i] * alt_weights_all[i][j]
                
        best_alternative = max(global_scores, key=global_scores.get)
        
        return render_template("index.html", criteria=criteria, alternatives=alternatives, results=True, 
                                      crit_weights=crit_weights, crit_consistency=crit_consistency,
                                      alt_weights=alt_weights_all, alt_consistency=alt_consistency_all,
                                      global_scores=global_scores, best_alternative=best_alternative)
    else:
        return render_template("index.html", criteria=criteria, alternatives=alternatives, results=None)

if __name__ == '__main__':
    app.run(debug=True)
