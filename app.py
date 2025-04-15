from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import pymysql
from datetime import datetime
import os 
import math 
import traceback 

app = Flask(__name__)
app.secret_key = os.urandom(24) 

CR_THRESHOLD = 0.10
RI_DICT = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
           7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48,
           13: 1.56, 14: 1.57, 15: 1.59}

@app.context_processor
def inject_global_constants():
    """Makes specified constants available to all templates."""
    return dict(
        RI_DICT=RI_DICT,
        CR_THRESHOLD=CR_THRESHOLD
    )

def get_connection():
    """Establishes a connection to the MySQL database."""
    try:
        conn = pymysql.connect(
            host="localhost",
            user="root",
            password="01020304", 
            database="ahp_danhnganh",
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor 
        )
        return conn
    except pymysql.Error as e:
        print(f"Database connection error: {e}")
        try:
            flash(f"Lỗi kết nối cơ sở dữ liệu: {e}", "error")
        except RuntimeError:
             print("Flash failed: Not in request context.")
        return None

def compute_pairwise_matrix(prefix, item_names, form):
    """
    Computes a pairwise comparison matrix from form data. (Robust version)

    Args:
        prefix (str): The prefix used for form input names (e.g., "pc", "alt_pc_1").
        item_names (list): A list of names for the items being compared.
        form (ImmutableMultiDict): The form data from the request.

    Returns:
        numpy.ndarray: The computed pairwise comparison matrix, or None if an error occurs.
    """
    n = len(item_names)
    if n <= 0:
        flash("Không thể tạo ma trận so sánh với 0 phần tử.", "error")
        return None
    matrix = np.ones((n, n), dtype=float) 

    for i in range(n):
        for j in range(i + 1, n):
            key = f"{prefix}_{i}_{j}"
            val_str = form.get(key)

            if val_str is None or val_str.strip() == "":
                flash(f"Thiếu giá trị so sánh giữa '{item_names[i]}' và '{item_names[j]}'. Vui lòng cung cấp tất cả các so sánh.", "error")
                return None 

            try:
                val = float(val_str)
                if val <= 0:
                    flash(f"Giá trị so sánh giữa '{item_names[i]}' và '{item_names[j]}' ({val_str}) phải là số dương.", "error")
                    return None
                if val < 1/9 - 1e-9 or val > 9 + 1e-9: 
                     print(f"Warning: Input value {val} for {key} ('{item_names[i]}' vs '{item_names[j]}') is outside the typical 1/9 to 9 range.")

                matrix[i, j] = val
                if val != 0:
                    matrix[j, i] = 1.0 / val
                else:
                    flash(f"Giá trị 0 không hợp lệ cho so sánh cặp.", "error")
                    return None

            except (ValueError, TypeError):
                 flash(f"Giá trị nhập vào '{val_str}' cho cặp ('{item_names[i]}', '{item_names[j]}') không hợp lệ. Vui lòng nhập một số.", "error")
                 return None 

    np.fill_diagonal(matrix, 1.0) 
    return matrix

def ahp_weighting(matrix):
    """
    Calculates weights (priority vector), lambda_max, CI, and CR for a pairwise
    comparison matrix using the eigenvector method.

    Args:
        matrix (numpy.ndarray): The pairwise comparison matrix.

    Returns:
        tuple: (weights, lambda_max, CI, CR, RI) or (None, None, None, None, None) if an error occurs.
               Weights are a 1D numpy array.
    """
    if matrix is None:
        return None, None, None, None, None

    n = matrix.shape[0]
    if n <= 0:
        return np.array([]), 0, 0, 0, 0 

    if np.any(matrix <= 0):
         flash("Ma trận chứa giá trị không dương. Không thể tính toán trọng số.", "error")
         return None, None, None, None, None

    try:
        eigvals, eigvecs = np.linalg.eig(matrix)

        real_eigvals = np.real(eigvals)
        max_eigval_idx = np.argmax(real_eigvals)
        lambda_max = real_eigvals[max_eigval_idx] 

        principal_eigvec = np.real(eigvecs[:, max_eigval_idx])

        if np.all(principal_eigvec <= 1e-9): 
            principal_eigvec = -principal_eigvec
        elif np.any(principal_eigvec < -1e-9): 
             flash("Cảnh báo: Vector trọng số chứa giá trị âm không mong muốn. Kết quả có thể không đáng tin cậy do tính không nhất quán cao.", "warning")

        sum_weights = np.sum(principal_eigvec)
        if abs(sum_weights) < 1e-9: 
             flash("Lỗi: Tổng vector trọng số gần bằng 0. Không thể chuẩn hóa.", "error")
             return None, None, None, None, None

        weights = principal_eigvec / sum_weights

        weights = np.maximum(0, weights)
        current_sum = np.sum(weights)
        if abs(current_sum - 1.0) > 1e-9 and current_sum > 1e-9:
            weights = weights / current_sum


        if n > 2: 
            CI = (lambda_max - n) / (n - 1)
            RI = RI_DICT.get(n, RI_DICT[max(RI_DICT.keys())])
            if RI == 0:
                 flash(f"Lỗi: Chỉ số ngẫu nhiên (RI) bằng 0 cho ma trận kích thước {n}.", "error")
                 CR = float('inf') 
            else:
                 if CI < 0 and CI > -1e-9:
                     CI = 0.0
                 elif CI < -1e-9:
                      flash(f"Cảnh báo: Chỉ số nhất quán (CI={CI:.4f}) có giá trị âm đáng kể. Có thể có lỗi tính toán hoặc ma trận đầu vào rất không nhất quán.", "warning")

                 CR = CI / RI
        elif n <= 2: 
            CI = 0.0
            RI = 0.00 
            CR = 0.0
        else:
             CI = None
             RI = None
             CR = None

        if any(math.isnan(x) or math.isinf(x) for x in [lambda_max, CI, CR] if x is not None) or \
           any(np.isnan(weights)) or any(np.isinf(weights)):
            flash("Lỗi: Kết quả tính toán chứa NaN hoặc vô cực. Kiểm tra ma trận đầu vào.", "error")
            print(f"NaN/Inf detected: lambda_max={lambda_max}, CI={CI}, CR={CR}, weights={weights}") 
            return None, None, None, None, None

        return weights, round(lambda_max, 5), round(CI, 5), round(CR, 5), RI

    except np.linalg.LinAlgError as e:
        flash(f"Lỗi tính toán trọng số (Lỗi Đại số tuyến tính: {e}). Ma trận có thể không hợp lệ hoặc không nhất quán nghiêm trọng.", "error")
        traceback.print_exc()
        return None, None, None, None, None
    except Exception as e:
        flash(f"Đã xảy ra lỗi không mong muốn trong quá trình tính toán AHP: {e}", "error")
        print(f"Unexpected AHP weighting error: {e}") 
        traceback.print_exc()
        return None, None, None, None, None


@app.route("/", methods=["GET"])
def index_redirect():
    """Redirects root URL to the first step."""
    return redirect(url_for('select_alternatives'))

@app.route("/select_alternatives", methods=["GET", "POST"])
def select_alternatives():
    """Step 0: User selects or enters alternatives."""
    if request.method == "POST":
        selection_mode = request.form.get('mode')
        alternatives = []
        all_db_alternatives = True # Flag to track if we can save to DB later

        if selection_mode == 'db':
            selected_ids_str = request.form.getlist('alternative_ids')
            if not selected_ids_str or len(selected_ids_str) < 2:
                flash("Vui lòng chọn ít nhất 2 phương án từ cơ sở dữ liệu.", "warning")
                return redirect(url_for('select_alternatives'))
            try:
                selected_ids = [int(id_str) for id_str in selected_ids_str]
            except ValueError:
                flash("ID phương án đã chọn không hợp lệ.", "error")
                return redirect(url_for('select_alternatives'))

            conn = get_connection()
            if not conn: return render_template("error.html", message="Lỗi kết nối DB.")
            try:
                with conn.cursor() as cursor:
                    format_strings = ','.join(['%s'] * len(selected_ids))
                    query = f"SELECT id, ten_phuong_an FROM phuong_an WHERE id IN ({format_strings}) ORDER BY FIELD(id, {format_strings})"
                    cursor.execute(query, tuple(selected_ids) * 2)
                    alternatives_from_db = cursor.fetchall()
                    if len(alternatives_from_db) != len(selected_ids):
                         raise ValueError("Không thể truy xuất tất cả các phương án đã chọn.")
                    alternatives = alternatives_from_db
            except (pymysql.Error, ValueError) as e:
                 flash(f"Lỗi truy xuất phương án từ DB: {e}", "error")
                 return redirect(url_for('select_alternatives'))
            finally:
                 if conn: conn.close()

        elif selection_mode == 'custom':
            custom_names = request.form.getlist('custom_alternative_names')
            unique_names = []
            seen_names = set()
            for name in custom_names:
                clean_name = name.strip()
                if clean_name and clean_name not in seen_names:
                    unique_names.append(clean_name)
                    seen_names.add(clean_name)

            if len(unique_names) < 2:
                flash("Vui lòng nhập ít nhất 2 tên phương án tùy chỉnh khác nhau và không trống.", "warning")
                return redirect(url_for('select_alternatives'))
            alternatives = [{'id': None, 'ten_phuong_an': name} for name in unique_names]
            all_db_alternatives = False
        else:
            flash("Vui lòng chọn chế độ 'Sử dụng Database' hoặc 'Nhập tùy chỉnh'.", "warning")
            return redirect(url_for('select_alternatives'))

        clear_session_data()
        session['session_alternatives'] = alternatives
        session['all_db_alternatives'] = all_db_alternatives
        session['alternatives_selected'] = True
        session.modified = True

        return redirect(url_for('select_criteria'))

    # --- GET Request ---
    clear_session_data()

    all_alternatives_db = []
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, ten_phuong_an FROM phuong_an ORDER BY id")
                all_alternatives_db = cursor.fetchall()
        except pymysql.Error as e:
             flash(f"Lỗi lấy danh sách phương án từ DB: {e}", "error")
        finally:
             if conn: conn.close()

    return render_template("select_alternatives.html", all_alternatives_db=all_alternatives_db)

@app.route("/select_criteria", methods=["GET", "POST"])
def select_criteria():
    """Step 1: User selects criteria."""
    if not session.get('alternatives_selected'):
        flash("Vui lòng chọn hoặc nhập các phương án trước.", "info")
        return redirect(url_for('select_alternatives'))

    if request.method == "POST":
        selected_ids_str = request.form.getlist('criteria_ids')
        if not selected_ids_str or len(selected_ids_str) < 2:
            flash("Vui lòng chọn ít nhất hai tiêu chí để so sánh.", "warning")
            return redirect(url_for('select_criteria'))

        try:
            selected_ids = [int(id_str) for id_str in selected_ids_str]
        except ValueError:
            flash("ID tiêu chí đã chọn không hợp lệ.", "error")
            return redirect(url_for('select_criteria'))

        conn = get_connection()
        if not conn: return render_template("error.html", message="Lỗi kết nối DB.")
        try:
            with conn.cursor() as cursor:
                format_strings = ','.join(['%s'] * len(selected_ids))
                query = f"SELECT id, ten_tieu_chi FROM tieu_chi WHERE id IN ({format_strings}) ORDER BY FIELD(id, {format_strings})"
                cursor.execute(query, tuple(selected_ids) * 2)
                selected_criteria = cursor.fetchall()
                if len(selected_criteria) != len(selected_ids):
                     raise ValueError("Could not retrieve all selected criteria.")
        except (pymysql.Error, ValueError) as e:
             flash(f"Lỗi truy xuất tiêu chí từ DB: {e}", "error")
             return redirect(url_for('select_criteria'))
        finally:
             if conn: conn.close()

        session['selected_criteria'] = selected_criteria
        session.pop('criteria_comparison_done', None)
        session.pop('alternative_comparisons_done', None)
        # ... (clear other related session keys as before) ...
        session.pop('current_alt_criterion_index', None)
        session.pop('alt_weights_all', None)
        session.pop('alt_matrices_all', None)
        session.pop('alt_lambda_max_all', None)
        session.pop('alt_ci_all', None)
        session.pop('alt_cr_all', None)
        session.pop('alt_ri_all', None)
        session.pop('final_scores', None)
        session.pop('best_alternative_info', None)
        clear_temporary_alt_data(len(selected_criteria))

        session['criteria_selected'] = True
        session.modified = True
        return redirect(url_for('compare_criteria'))

    # --- GET request ---
    all_criteria = []
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, ten_tieu_chi FROM tieu_chi ORDER BY id")
                all_criteria = cursor.fetchall()
        except pymysql.Error as e:
             flash(f"Lỗi lấy danh sách tiêu chí từ DB: {e}", "error")
        finally:
             if conn: conn.close()

    return render_template("select_criteria.html",
                           all_criteria=all_criteria,
                           selected_alternatives=session.get('session_alternatives', []))

@app.route("/compare_criteria", methods=["GET", "POST"])
def compare_criteria():
    """Step 2: User compares selected criteria."""
    if not session.get('criteria_selected'):
        flash("Vui lòng chọn tiêu chí trước.", "info")
        return redirect(url_for('select_criteria'))

    selected_criteria = session.get('selected_criteria', [])
    if not selected_criteria:
        flash("Không tìm thấy tiêu chí đã chọn trong session. Vui lòng chọn lại.", "error")
        return redirect(url_for('select_criteria'))

    criteria_names = [c['ten_tieu_chi'] for c in selected_criteria]
    num_criteria = len(selected_criteria)

    if request.method == "POST":
        crit_matrix = compute_pairwise_matrix("pc", criteria_names, request.form)

        if crit_matrix is None:
            session['form_data_crit'] = request.form
            session.modified = True
            session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
            return redirect(url_for('compare_criteria'))

        # Use the NEW ahp_weighting function
        crit_weights, crit_lambda_max, crit_ci, crit_cr, crit_ri = ahp_weighting(crit_matrix)

        if crit_weights is None:
            session['form_data_crit'] = request.form
            session.modified = True
            session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
            return redirect(url_for('compare_criteria'))

        session['crit_matrix'] = crit_matrix.tolist()
        session['crit_weights'] = crit_weights.tolist()
        session['crit_lambda_max'] = crit_lambda_max
        session['crit_ci'] = crit_ci
        session['crit_cr'] = crit_cr
        session['crit_ri'] = crit_ri

        if crit_cr > CR_THRESHOLD:
            flash(f"Tỷ số nhất quán (CR = {crit_cr:.4f}) vượt ngưỡng cho phép ({CR_THRESHOLD:.2f}). Vui lòng xem lại các so sánh tiêu chí.", "error")
            session['form_data_crit'] = request.form
            session['criteria_comparison_done'] = False
            session.modified = True
            return redirect(url_for('compare_criteria'))

        flash(f"So sánh tiêu chí thành công (CR = {crit_cr:.4f}). Tiếp tục so sánh phương án.", "success")
        session['criteria_comparison_done'] = True
        session.pop('form_data_crit', None)

        num_alternatives = len(session.get('session_alternatives', []))
        session['alt_matrices_all'] = [[None for _ in range(num_alternatives)] for _ in range(num_criteria)]
        session['alt_weights_all'] = [None] * num_criteria
        session['alt_lambda_max_all'] = [None] * num_criteria
        session['alt_ci_all'] = [None] * num_criteria
        session['alt_cr_all'] = [None] * num_criteria
        session['alt_ri_all'] = [None] * num_criteria
        session['current_alt_criterion_index'] = 0
        session.modified = True

        return redirect(url_for('compare_alternatives'))

    # --- GET request ---
    form_data = session.get('form_data_crit', None)
    crit_lambda_max = session.get('crit_lambda_max')
    crit_ci = session.get('crit_ci')
    crit_cr = session.get('crit_cr')
    crit_ri = session.get('crit_ri')

    return render_template("compare_criteria.html",
                           criteria=selected_criteria,
                           criteria_names=criteria_names,
                           form_data=form_data,
                           crit_lambda_max=crit_lambda_max,
                           crit_ci=crit_ci,
                           crit_cr=crit_cr,
                           crit_ri=crit_ri)

@app.route("/compare_alternatives", methods=["GET", "POST"])
def compare_alternatives():
    """Step 3: User compares alternatives for each criterion iteratively."""
    if not session.get('criteria_comparison_done'):
        flash("Vui lòng hoàn thành so sánh tiêu chí (với CR hợp lệ) trước.", "info")
        return redirect(url_for('compare_criteria'))
    if 'session_alternatives' not in session or 'selected_criteria' not in session:
        flash("Dữ liệu session bị thiếu hoặc không hợp lệ (phương án/tiêu chí). Vui lòng bắt đầu lại.", "error")
        return redirect(url_for('select_alternatives'))

    selected_criteria = session['selected_criteria']
    alternatives = session['session_alternatives']
    alternative_names = [a['ten_phuong_an'] for a in alternatives]
    num_alternatives = len(alternatives)
    num_criteria = len(selected_criteria)
    current_index = session.get('current_alt_criterion_index', 0)

    if current_index >= num_criteria:
        session['alternative_comparisons_done'] = True
        session.modified = True
        flash("Tất cả so sánh phương án đã hoàn thành.", "info")
        return redirect(url_for('calculate_results'))

    current_criterion = selected_criteria[current_index]

    if request.method == "POST":
        prefix = f"alt_pc_{current_criterion['id']}"
        alt_matrix = compute_pairwise_matrix(prefix, alternative_names, request.form)

        if alt_matrix is None:
            session['form_data_alt'] = request.form
            session.pop(f'temp_alt_lambda_max_{current_index}', None); session.pop(f'temp_alt_ci_{current_index}', None); session.pop(f'temp_alt_cr_{current_index}', None); session.pop(f'temp_alt_ri_{current_index}', None)
            session.modified = True
            return redirect(url_for('compare_alternatives'))

        # Use the NEW ahp_weighting function
        alt_weights, alt_lambda_max, alt_ci, alt_cr, alt_ri = ahp_weighting(alt_matrix)

        if alt_weights is None:
            session['form_data_alt'] = request.form
            session.pop(f'temp_alt_lambda_max_{current_index}', None); session.pop(f'temp_alt_ci_{current_index}', None); session.pop(f'temp_alt_cr_{current_index}', None); session.pop(f'temp_alt_ri_{current_index}', None)
            session.modified = True
            return redirect(url_for('compare_alternatives'))

        session[f'temp_alt_matrix_{current_index}'] = alt_matrix.tolist()
        session[f'temp_alt_lambda_max_{current_index}'] = alt_lambda_max
        session[f'temp_alt_ci_{current_index}'] = alt_ci
        session[f'temp_alt_cr_{current_index}'] = alt_cr
        session[f'temp_alt_ri_{current_index}'] = alt_ri

        if alt_cr > CR_THRESHOLD:
            flash(f"Tỷ số nhất quán cho phương án theo tiêu chí '{current_criterion['ten_tieu_chi']}' (CR = {alt_cr:.4f}) vượt ngưỡng ({CR_THRESHOLD:.2f}). Vui lòng xem lại.", "error")
            session['form_data_alt'] = request.form
            session.modified = True
            return redirect(url_for('compare_alternatives'))

        # --- Consistent! Store results permanently ---
        def ensure_session_list(key, length, default_val=None):
            if key not in session or not isinstance(session[key], list) or len(session[key]) != length:
                session[key] = [default_val] * length
            elif len(session[key]) != length:
                 session[key] = [default_val] * length
            return session[key]

        alt_matrices_all = ensure_session_list('alt_matrices_all', num_criteria, default_val=[])
        alt_weights_all = ensure_session_list('alt_weights_all', num_criteria)
        alt_lambda_max_all = ensure_session_list('alt_lambda_max_all', num_criteria)
        alt_ci_all = ensure_session_list('alt_ci_all', num_criteria)
        alt_cr_all = ensure_session_list('alt_cr_all', num_criteria)
        alt_ri_all = ensure_session_list('alt_ri_all', num_criteria)

        alt_matrices_all[current_index] = alt_matrix.tolist()
        alt_weights_all[current_index] = alt_weights.tolist()
        alt_lambda_max_all[current_index] = alt_lambda_max
        alt_ci_all[current_index] = alt_ci
        alt_cr_all[current_index] = alt_cr
        alt_ri_all[current_index] = alt_ri

        session['alt_matrices_all'] = alt_matrices_all
        session['alt_weights_all'] = alt_weights_all
        session['alt_lambda_max_all'] = alt_lambda_max_all
        session['alt_ci_all'] = alt_ci_all
        session['alt_cr_all'] = alt_cr_all
        session['alt_ri_all'] = alt_ri_all

        # Clear temporary data
        session.pop(f'temp_alt_matrix_{current_index}', None); session.pop(f'temp_alt_lambda_max_{current_index}', None); session.pop(f'temp_alt_ci_{current_index}', None); session.pop(f'temp_alt_cr_{current_index}', None); session.pop(f'temp_alt_ri_{current_index}', None)
        session.pop('form_data_alt', None)

        next_index = current_index + 1
        session['current_alt_criterion_index'] = next_index
        flash(f"So sánh phương án theo '{current_criterion['ten_tieu_chi']}' đã lưu (CR = {alt_cr:.4f}).", "success")
        session.modified = True

        if next_index >= num_criteria:
            session['alternative_comparisons_done'] = True
            return redirect(url_for('calculate_results'))
        else:
            return redirect(url_for('compare_alternatives'))

    # --- GET request ---
    form_data = session.get('form_data_alt', None)
    alt_lambda_max = session.get(f'temp_alt_lambda_max_{current_index}')
    alt_ci = session.get(f'temp_alt_ci_{current_index}')
    alt_cr = session.get(f'temp_alt_cr_{current_index}')
    alt_ri = session.get(f'temp_alt_ri_{current_index}')

    if 'form_data_alt' in session:
        session.pop('form_data_alt')
        session.modified = True

    return render_template("compare_alternatives.html",
                           criterion=current_criterion,
                           alternatives=alternatives,
                           alternative_names=alternative_names,
                           form_data=form_data,
                           alt_lambda_max=alt_lambda_max,
                           alt_ci=alt_ci,
                           alt_cr=alt_cr,
                           alt_ri=alt_ri,
                           criterion_index=current_index,
                           total_criteria=num_criteria)

@app.route("/calculate_results")
def calculate_results():
    """Step 4: Calculate final scores and display detailed results."""
    if not session.get('criteria_comparison_done'):
        flash("Bước so sánh tiêu chí chưa hoàn thành hoặc session đã hết hạn.", "warning")
        return redirect(url_for('compare_criteria'))

    num_criteria = len(session.get('selected_criteria', []))
    num_alternatives = len(session.get('session_alternatives', []))

    if not session.get('alternative_comparisons_done'):
         current_alt_index = session.get('current_alt_criterion_index', -1)
         if current_alt_index == num_criteria and num_criteria > 0:
              session['alternative_comparisons_done'] = True
              session.modified = True
         else:
              flash("Bước so sánh phương án chưa hoàn thành đầy đủ hoặc session đã hết hạn.", "warning")
              return redirect(url_for('compare_alternatives'))

    # --- Validate required session data ---
    required_keys = ['crit_weights', 'alt_weights_all', 'session_alternatives', 'selected_criteria',
                     'crit_matrix', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri',
                     'alt_matrices_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all']
    missing_or_invalid = []
    # (Validation logic remains the same as in the first app.py)
    for key in required_keys:
        data = session.get(key)
        if data is None:
            missing_or_invalid.append(f"'{key}' is missing")
            continue
        if key in ['alt_weights_all', 'alt_matrices_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all']:
             if not isinstance(data, list) or len(data) != num_criteria:
                 missing_or_invalid.append(f"'{key}' is not a list or has incorrect length (expected {num_criteria})")
             elif any(x is None for x in data):
                 try:
                     missing_indices = [i for i, x in enumerate(data) if x is None]
                     criteria_names = [session['selected_criteria'][i]['ten_tieu_chi'] for i in missing_indices]
                     missing_or_invalid.append(f"'{key}' has missing data for criteria: {', '.join(criteria_names)}")
                 except (IndexError, KeyError):
                      missing_or_invalid.append(f"'{key}' has missing data and criteria info is inconsistent.")

    if missing_or_invalid:
        error_message = "Dữ liệu session không đầy đủ hoặc không hợp lệ: " + "; ".join(missing_or_invalid) + ". Vui lòng quay lại các bước trước hoặc bắt đầu lại."
        flash(error_message, "error")
        if any('alt_' in s for s in missing_or_invalid): return redirect(url_for('compare_alternatives'))
        elif any('crit_' in s for s in missing_or_invalid): return redirect(url_for('compare_criteria'))
        else: return redirect(url_for('clear_session_and_start'))

    # --- Perform Final Calculation ---
    try:
        crit_weights = np.array(session['crit_weights'])
        alt_weights_all_list = session['alt_weights_all']
        alternatives = session['session_alternatives']
        selected_criteria = session['selected_criteria']

        alt_weights_matrix = np.array(alt_weights_all_list) 

        if crit_weights.shape != (num_criteria,): raise ValueError(f"Kích thước trọng số tiêu chí ({crit_weights.shape}) không khớp ({num_criteria}).")
        if alt_weights_matrix.shape != (num_criteria, num_alternatives): raise ValueError(f"Kích thước ma trận trọng số PA ({alt_weights_matrix.shape}) không khớp ({num_criteria}, {num_alternatives}).")
        if np.isnan(crit_weights).any() or np.isinf(crit_weights).any(): raise ValueError("NaN/Inf trong trọng số tiêu chí.")
        if np.isnan(alt_weights_matrix).any() or np.isinf(alt_weights_matrix).any(): raise ValueError("NaN/Inf trong trọng số phương án.")
        if abs(np.sum(crit_weights) - 1.0) > 1e-6: flash(f"Cảnh báo: Tổng trọng số tiêu chí không gần bằng 1 ({np.sum(crit_weights):.6f}).", "warning")
        for i in range(alt_weights_matrix.shape[0]):
             row_sum = np.sum(alt_weights_matrix[i, :])
             if abs(row_sum - 1.0) > 1e-6: flash(f"Cảnh báo: Tổng trọng số PA cho tiêu chí '{selected_criteria[i]['ten_tieu_chi']}' không gần bằng 1 ({row_sum:.6f}).", "warning")

        final_scores_vector = np.dot(crit_weights, alt_weights_matrix)

        if final_scores_vector.shape != (num_alternatives,): raise ValueError(f"Kích thước vector điểm tổng hợp ({final_scores_vector.shape}) không khớp ({num_alternatives}).")
        if abs(np.sum(final_scores_vector) - 1.0) > 1e-6: flash(f"Cảnh báo: Tổng điểm cuối cùng không gần bằng 1 ({np.sum(final_scores_vector):.6f}).", "warning")

        final_scores_dict = {alt['ten_phuong_an']: score for alt, score in zip(alternatives, final_scores_vector)}
        best_alternative_name = max(final_scores_dict, key=final_scores_dict.get) if final_scores_dict else None
        best_alternative_info = None
        if best_alternative_name:
             best_alt_match = next((alt for alt in alternatives if alt['ten_phuong_an'] == best_alternative_name), None)
             if best_alt_match:
                best_alternative_info = {'id': best_alt_match.get('id'), 'name': best_alternative_name, 'score': final_scores_dict[best_alternative_name]}

    except (ValueError, TypeError, IndexError) as e:
         flash(f"Lỗi trong quá trình tính toán cuối cùng: {e}. Dữ liệu có thể bị lỗi.", "error")
         print(f"Calculation Error Details: {e}"); traceback.print_exc()
         return render_template("error.html", message=f"Lỗi tính toán: {e}")
    except Exception as e:
         flash(f"Đã xảy ra lỗi không mong muốn trong quá trình tính toán cuối cùng: {e}.", "error")
         print(f"Unexpected Calculation Error: {e}"); traceback.print_exc()
         return render_template("error.html", message=f"Lỗi không mong muốn: {e}")

    # --- Prepare data for display (same as before) ---
    results_display = []
    if alternatives and final_scores_dict:
        for i, alt in enumerate(alternatives):
            score = final_scores_vector[i]
            results_display.append({'id': alt.get('id'), 'name': alt['ten_phuong_an'], 'score': score, 'is_best': alt['ten_phuong_an'] == best_alternative_name if best_alternative_name else False})
        results_display.sort(key=lambda x: x['score'], reverse=True)

    session['final_scores'] = final_scores_dict
    session['best_alternative_info'] = best_alternative_info
    session.modified = True

    # --- Save results to database (same logic as before) ---
    can_save_to_db = session.get('all_db_alternatives', False)
    save_attempted = False
    save_successful = False
    if can_save_to_db and results_display:
        save_attempted = True
        conn = get_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    timestamp = datetime.now()
                    conn.begin()
                    insert_query = "INSERT INTO ket_qua (id_phuong_an, diem_tong_hop, la_tot_nhat, thoi_gian) VALUES (%s, %s, %s, %s)"
                    values_to_insert = []
                    for result in results_display:
                         if result['id'] is not None:
                             values_to_insert.append((result['id'], result['score'], result['is_best'], timestamp))
                         else: print(f"Warning: Skipping save for result with missing ID: {result['name']}")
                    if values_to_insert:
                        cursor.executemany(insert_query, values_to_insert)
                        conn.commit()
                        save_successful = True
                        flash("Kết quả cuối cùng đã được tính toán và lưu vào cơ sở dữ liệu.", "success")
                    else: flash("Không có dữ liệu hợp lệ để lưu vào cơ sở dữ liệu (thiếu ID phương án).", "warning")
            except pymysql.Error as e:
                conn.rollback(); flash(f"Lỗi lưu kết quả vào cơ sở dữ liệu: {e}", "error"); print(f"DB Save Error: {e}"); traceback.print_exc()
            except Exception as e:
                conn.rollback(); flash(f"Đã xảy ra lỗi không mong muốn khi lưu kết quả: {e}", "error"); print(f"Unexpected DB Save Error: {e}"); traceback.print_exc()
            finally:
                if conn: conn.close()
        else: flash("Không thể kết nối đến cơ sở dữ liệu để lưu kết quả.", "error")
    elif not can_save_to_db: flash("Kết quả được tính toán nhưng không được lưu vào cơ sở dữ liệu vì có sử dụng phương án tùy chỉnh.", "info")
    elif not results_display: flash("Không có kết quả để lưu vào cơ sở dữ liệu.", "info")

    # --- Prepare Intermediate Results for Display (same as before) ---
    intermediate_results = {}
    try:
        intermediate_results = {
            'criteria': session.get('selected_criteria', []), 'crit_matrix': session.get('crit_matrix'), 'crit_weights': session.get('crit_weights'),
            'crit_lambda_max': session.get('crit_lambda_max'), 'crit_ci': session.get('crit_ci'), 'crit_cr': session.get('crit_cr'), 'crit_ri': session.get('crit_ri'),
            'alternatives': session.get('session_alternatives', []), 'alt_matrices_all': session.get('alt_matrices_all'), 'alt_weights_all': session.get('alt_weights_all'),
            'alt_lambda_max_all': session.get('alt_lambda_max_all'), 'alt_ci_all': session.get('alt_ci_all'), 'alt_cr_all': session.get('alt_cr_all'), 'alt_ri_all': session.get('alt_ri_all'),
        }
        num_crit_check = len(intermediate_results.get('criteria', []))
        if len(intermediate_results.get('alt_matrices_all', [])) != num_crit_check: raise ValueError(f"Intermediate alternative matrices count mismatch (expected {num_crit_check}).")
        if len(intermediate_results.get('alt_weights_all', [])) != num_crit_check: raise ValueError(f"Intermediate alternative weights count mismatch (expected {num_crit_check}).")
    except Exception as e:
         flash(f"Lỗi khi chuẩn bị dữ liệu trung gian để hiển thị: {e}", "warning"); print(f"Error preparing intermediate results: {e}"); traceback.print_exc(); intermediate_results = {}

    return render_template("results.html",
                           results=results_display, intermediate=intermediate_results, best_alternative_info=best_alternative_info,
                           save_attempted=save_attempted, save_successful=save_successful, can_save_to_db=can_save_to_db)

@app.route("/results_history")
def results_history():
    """Displays recent results from the database, grouped by analysis run time."""
    # (This function remains the same as in the first app.py)
    conn = get_connection()
    grouped_history = {}
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT DISTINCT thoi_gian FROM ket_qua ORDER BY thoi_gian DESC LIMIT 20")
                timestamps = cursor.fetchall()
                for ts_row in timestamps:
                    ts = ts_row['thoi_gian']
                    cursor.execute("SELECT p.ten_phuong_an, k.diem_tong_hop, k.la_tot_nhat, k.thoi_gian FROM ket_qua k JOIN phuong_an p ON k.id_phuong_an = p.id WHERE k.thoi_gian = %s ORDER BY k.diem_tong_hop DESC", (ts,))
                    entries_for_ts = cursor.fetchall()
                    if entries_for_ts:
                        grouped_history[ts] = {'timestamp_obj': ts, 'timestamp_str': ts.strftime('%Y-%m-%d %H:%M:%S'), 'results': entries_for_ts}
        except pymysql.Error as e: flash(f"Lỗi lấy lịch sử kết quả: {e}", "error"); print(f"DB History Error: {e}"); traceback.print_exc()
        except Exception as e: flash(f"Đã xảy ra lỗi không mong muốn khi lấy lịch sử: {e}", "error"); print(f"Unexpected History Error: {e}"); traceback.print_exc()
        finally:
            if conn: conn.close()
    else: pass # Error flashed by get_connection

    sorted_history_list = sorted(grouped_history.values(), key=lambda item: item['timestamp_obj'], reverse=True)
    return render_template("results_history.html", history_list=sorted_history_list)

# --- Helper Functions (Unchanged from the first app.py) ---
def clear_temporary_alt_data(num_criteria):
     """Clears temporary session keys used during alternative comparison retries."""
     max_crit_guess = max(num_criteria if num_criteria else 0, 20)
     for i in range(max_crit_guess):
         session.pop(f'temp_alt_matrix_{i}', None); session.pop(f'temp_alt_lambda_max_{i}', None); session.pop(f'temp_alt_ci_{i}', None); session.pop(f'temp_alt_cr_{i}', None); session.pop(f'temp_alt_ri_{i}', None)
     session.pop('form_data_alt', None)

def clear_session_data():
    """Clears specific session keys related to an AHP run more comprehensively."""
    keys_to_clear = [
        'session_alternatives', 'all_db_alternatives', 'alternatives_selected', 'selected_criteria', 'criteria_selected',
        'crit_matrix', 'crit_weights', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri', 'criteria_comparison_done', 'form_data_crit',
        'alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all',
        'current_alt_criterion_index', 'alternative_comparisons_done', 'final_scores', 'best_alternative_info'
    ]
    for key in keys_to_clear:
        session.pop(key, None)
    num_crit = len(session.get('selected_criteria', []))
    clear_temporary_alt_data(num_crit)
    session.modified = True

@app.route("/clear")
def clear_session_and_start():
    """Clears the session and redirects to the start (select alternatives)."""
    clear_session_data()
    flash("Session đã được xóa. Bắt đầu một phân tích mới.", "info")
    return redirect(url_for('select_alternatives'))

@app.errorhandler(404)
def page_not_found(e):
     flash("Trang yêu cầu không được tìm thấy (404).", "error")
     return render_template('error.html', message='Trang không tìm thấy (404)'), 404

@app.errorhandler(500)
def internal_server_error(e):
     print(f"Internal Server Error: {e}"); traceback.print_exc()
     flash("Đã xảy ra lỗi máy chủ nội bộ (500). Vui lòng thử lại sau hoặc liên hệ quản trị viên.", "error")
     return render_template('error.html', message='Lỗi Máy chủ Nội bộ (500)'), 500

if __name__ == "__main__":
    conn_test = get_connection()
    if conn_test is None:
         print("\n*** CẢNH BÁO: Không thể kết nối đến cơ sở dữ liệu! ***")
         print("Vui lòng kiểm tra thông tin đăng nhập (đặc biệt là mật khẩu) trong hàm get_connection().\n")
    else:
        conn_test.close()
    app.run(debug=True)

