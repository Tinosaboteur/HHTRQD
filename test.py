from flask import Flask, request, render_template_string
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

# Template HTML
template = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Hệ hỗ trợ ra quyết định theo AHP</title>
  <style>
    table, th, td { border: 1px solid #888; border-collapse: collapse; padding: 5px; text-align: center; }
    th { background-color: #eee; }
    input { width: 80px; }
    h2, h3, h4 { margin-bottom: 5px; }
  </style>
</head>
<body>
  <h1>Hệ hỗ trợ ra quyết định chọn ngành học (Phương pháp AHP)</h1>
  {% if not results %}
  <form method="post">
    <!-- Phần I: Ma trận so sánh cặp các tiêu chí -->
    <h2>I. Ma trận so sánh cặp các tiêu chí</h2>
    <p>Nhập giá trị cho các ô ở phần trên tam giác (i &lt; j). Các ô đường chéo cố định là 1, ô dưới tam giác sẽ được tính tự động.</p>
    <table>
      <tr>
        <th>Tiêu chí</th>
        {% for j, crit in enumerate(criteria) %}
          <th>{{ crit }}</th>
        {% endfor %}
      </tr>
      {% for i, crit in enumerate(criteria) %}
      <tr>
        <th>{{ crit }}</th>
        {% for j in range(criteria|length) %}
          {% if i == j %}
            <td>1</td>
          {% elif i < j %}
            <td><input type="number" step="any" name="pc_{{ i }}_{{ j }}" required></td>
          {% else %}
            <td>--</td>
          {% endif %}
        {% endfor %}
      </tr>
      {% endfor %}
    </table>
    <br>
    
    <!-- Phần II: Ma trận so sánh cặp các phương án cho mỗi tiêu chí -->
    <h2>II. Ma trận so sánh cặp các phương án theo từng tiêu chí</h2>
    {% for i, crit in enumerate(criteria) %}
      <h3>Tiêu chí: {{ crit }}</h3>
      <p>Nhập giá trị cho các ô ở phần trên tam giác (i &lt; j). Đường chéo cố định là 1, ô dưới tam giác sẽ được tính tự động.</p>
      <table>
        <tr>
          <th>Phương án</th>
          {% for j, alt in enumerate(alternatives) %}
            <th>{{ alt }}</th>
          {% endfor %}
        </tr>
        {% for r, alt_r in enumerate(alternatives) %}
        <tr>
          <th>{{ alt_r }}</th>
          {% for c in range(alternatives|length) %}
            {% if r == c %}
              <td>1</td>
            {% elif r < c %}
              <td><input type="number" step="any" name="alt_pc_{{ i }}_{{ r }}_{{ c }}" required></td>
            {% else %}
              <td>--</td>
            {% endif %}
          {% endfor %}
        </tr>
        {% endfor %}
      </table>
      <br>
    {% endfor %}
    <input type="submit" value="Tính toán">
  </form>
  {% else %}
  <!-- Kết quả -->
  <h2>Kết quả tính toán</h2>
  <h3>A. Trọng số tiêu chí (tính từ ma trận so sánh cặp các tiêu chí)</h3>
  <table>
    <tr>
      <th>Tiêu chí</th>
      <th>Trọng số</th>
    </tr>
    {% for i, crit in enumerate(criteria) %}
    <tr>
      <td>{{ crit }}</td>
      <td>{{ crit_weights[i] | round(4) }}</td>
    </tr>
    {% endfor %}
  </table>
  {% if crit_consistency %}
  <h4>Kiểm tra tính nhất quán (CR của tiêu chí): {{ crit_consistency.cr | round(4) }}
    {% if crit_consistency.cr < 0.1 %}
      (Chấp nhận)
    {% else %}
      (Không chấp nhận)
    {% endif %}
  </h4>
  {% endif %}
  
  <h3>B. Trọng số các phương án theo từng tiêu chí</h3>
  {% for i, crit in enumerate(criteria) %}
    <h4>Tiêu chí: {{ crit }}</h4>
    <table>
      <tr>
        <th>Phương án</th>
        <th>Trọng số</th>
      </tr>
      {% for j, alt in enumerate(alternatives) %}
      <tr>
        <td>{{ alt }}</td>
        <td>{{ alt_weights[i][j] | round(4) }}</td>
      </tr>
      {% endfor %}
    </table>
    {% if alt_consistency and alt_consistency[i] %}
    <h5>CR (phương án) cho tiêu chí này: {{ alt_consistency[i].cr | round(4) }}
      {% if alt_consistency[i].cr < 0.1 %}
        (Chấp nhận)
      {% else %}
        (Không chấp nhận)
      {% endif %}
    </h5>
    {% endif %}
    <br>
  {% endfor %}
  
  <h3>C. Điểm tổng hợp của các phương án</h3>
  <table>
    <tr>
      <th>Phương án</th>
      <th>Điểm tổng hợp</th>
    </tr>
    {% for alt, score in global_scores.items() %}
    <tr>
      <td>{{ alt }}</td>
      <td>{{ score | round(4) }}</td>
    </tr>
    {% endfor %}
  </table>
  <h3>Phương án được chọn: <span style="color:green;">{{ best_alternative }}</span></h3>
  <br>
  <a href="/">Thực hiện lại</a>
  {% endif %}
</body>
</html>
"""

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
        
        return render_template_string(template, criteria=criteria, alternatives=alternatives, results=True, 
                                      crit_weights=crit_weights, crit_consistency=crit_consistency,
                                      alt_weights=alt_weights_all, alt_consistency=alt_consistency_all,
                                      global_scores=global_scores, best_alternative=best_alternative, enumerate=enumerate)
    else:
        return render_template_string(template, criteria=criteria, alternatives=alternatives, results=None, enumerate=enumerate)

if __name__ == '__main__':
    app.run(debug=True)
