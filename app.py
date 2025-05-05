# -*- coding: utf-8 -*-
from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import psycopg2 # <--- Import psycopg2
import psycopg2.extras # <--- For DictCursor
from datetime import datetime
import os
import math
import traceback
import json
import pandas as pd
from werkzeug.utils import secure_filename
import uuid
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
# Load secret key from environment variable for security
app.secret_key = os.environ.get('SECRET_KEY', 'default_fallback_secret_key_for_dev_only') # Use env var
app.config['UPLOAD_FOLDER'] = 'uploads'
# Limit upload size to 1MB
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 # 1 MB

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

MIN_CRITERIA = 4
MIN_ALTERNATIVES = 4
CR_THRESHOLD = 0.10
RI_DICT = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
           7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48,
           13: 1.56, 14: 1.57, 15: 1.59}

@app.context_processor
def inject_global_constants():
    """Makes specified constants available to all templates."""
    return dict(
        RI_DICT=RI_DICT,
        CR_THRESHOLD=CR_THRESHOLD,
        MIN_CRITERIA=MIN_CRITERIA,
        MIN_ALTERNATIVES=MIN_ALTERNATIVES
    )

def get_connection():
    """Establishes a connection to the PostgreSQL database using DATABASE_URL."""
    conn = None
    database_url = os.environ.get('DATABASE_URL', None) # Lấy DATABASE_URL từ env
    if not database_url:
        print("Database connection error: DATABASE_URL environment variable not set.")
        flash("Lỗi cấu hình: Không tìm thấy chuỗi kết nối cơ sở dữ liệu.", "error")
        return None
    try:
        conn = psycopg2.connect(database_url) # Use DATABASE_URL
        return conn
    except psycopg2.OperationalError as e:
        # Lỗi cụ thể hơn khi không kết nối được (sai host, port, db name, network issue)
        print(f"Database connection error (Operational): {e}")
        flash(f"Lỗi kết nối Database: Không thể kết nối tới server. Kiểm tra lại chuỗi kết nối và trạng thái DB.", "error")
        return None
    except psycopg2.Error as e:
        # Các lỗi psycopg2 khác (vd: authentication)
        print(f"Database connection error: {e}")
        flash(f"Lỗi Database: {e}", "error")
        return None
    except Exception as e:
        print(f"Unexpected error getting connection: {e}")
        flash(f"Lỗi không mong muốn khi kết nối DB: {e}", "error")
        return None

def execute_query(query, params=None, fetchone=False, fetchall=False, commit=False):
    """ Executes a query with error handling and connection management. """
    conn = get_connection()
    if not conn:
        # get_connection đã flash lỗi rồi
        return None

    result = None
    try:
        # Use DictCursor to get results as dictionaries
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(query, params)
            if fetchone:
                result = cursor.fetchone()
                if result is not None:
                    result = dict(result) # Convert DictRow to dict
            elif fetchall:
                result = cursor.fetchall()
                if result is not None:
                    result = [dict(row) for row in result] # Convert list of DictRow to list of dict

            if commit:
                conn.commit()

    except psycopg2.Error as e:
        if conn: conn.rollback() # Rollback on error if transaction started
        print(f"Database query error: {e}")
        print(f"Query: {query}")
        print(f"Params: {params}")
        traceback.print_exc()
        flash(f"Lỗi truy vấn cơ sở dữ liệu: {e}", "error")
        result = None # Ensure result is None on error
    except Exception as e:
        if conn: conn.rollback()
        print(f"Unexpected error during query execution: {e}")
        traceback.print_exc()
        flash(f"Lỗi không mong muốn khi thực thi truy vấn: {e}", "error")
        result = None
    finally:
        if conn:
            conn.close()
    return result

# --- compute_pairwise_matrix, parse_excel_matrix, ahp_weighting ---
# (Giữ nguyên các hàm này như phiên bản trước, chúng hoạt động tốt)
def compute_pairwise_matrix(prefix, item_names, form):
    """Computes a pairwise comparison matrix from form data."""
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

                matrix[i, j] = val
                if abs(val) > 1e-9:
                    matrix[j, i] = 1.0 / val
                else:
                    flash(f"Giá trị 0 không hợp lệ cho so sánh cặp.", "error")
                    return None

            except (ValueError, TypeError):
                 flash(f"Giá trị nhập vào '{val_str}' cho cặp ('{item_names[i]}', '{item_names[j]}') không hợp lệ. Vui lòng nhập một số.", "error")
                 return None

    np.fill_diagonal(matrix, 1.0)
    return matrix

def parse_excel_matrix(file_storage, expected_size, item_names_for_validation=None):
    """Parses a pairwise matrix from an Excel file."""
    if not file_storage or file_storage.filename == '':
        return None, "Không có file nào được chọn."
    allowed_extensions = ('.xlsx', '.xls', '.xlsm', '.xlsb')
    if not file_storage.filename.lower().endswith(allowed_extensions):
        return None, "Định dạng file không hợp lệ. Chỉ chấp nhận file Excel (ví dụ: .xlsx, .xls)."

    try:
        df = pd.read_excel(file_storage, header=None, engine='openpyxl')
        start_row, start_col = -1, -1
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                cell_value = df.iloc[r, c]
                is_numeric = isinstance(cell_value, (int, float, np.number))
                is_valid_numeric = is_numeric and not pd.isna(cell_value)
                if is_valid_numeric:
                    if abs(float(cell_value) - 1.0) < 1e-6:
                        if r + 1 < df.shape[0] and c + 1 < df.shape[1]:
                             next_cell = df.iloc[r+1, c+1]
                             if isinstance(next_cell, (int, float, np.number)) and not pd.isna(next_cell):
                                start_row, start_col = r, c
                                break
            if start_row != -1: break

        if start_row == -1 or start_col == -1:
            return None, "Không thể tự động xác định ma trận số trong file Excel. Đảm bảo ma trận bắt đầu bằng số 1 ở góc trên bên trái và chỉ chứa các giá trị số hợp lệ."

        if start_row + expected_size > df.shape[0] or start_col + expected_size > df.shape[1]:
            return None, f"Kích thước ma trận số tìm thấy không đủ lớn. Cần ma trận {expected_size}x{expected_size} bắt đầu từ ô ({start_row+1},{start_col+1})."

        matrix_df = df.iloc[start_row : start_row + expected_size, start_col : start_col + expected_size]
        matrix_np = matrix_df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

        if np.isnan(matrix_np).any():
            nan_loc = np.argwhere(np.isnan(matrix_np))
            first_nan_row, first_nan_col = nan_loc[0]
            return None, f"Ma trận chứa giá trị không phải số hoặc ô trống tại vị trí ({start_row + first_nan_row + 1},{start_col + first_nan_col + 1}) trong file Excel."

        if matrix_np.shape != (expected_size, expected_size):
            return None, f"Kích thước ma trận không đúng. Cần {expected_size}x{expected_size}, tìm thấy {matrix_np.shape}."

        if not np.allclose(np.diag(matrix_np), 1.0):
            diag_diff = np.where(np.abs(np.diag(matrix_np) - 1.0) > 1e-6)[0]
            diff_indices = [start_row + i + 1 for i in diag_diff]
            return None, f"Đường chéo chính của ma trận phải bằng 1. Lỗi tại hàng/cột tương ứng: {diff_indices}."

        for i in range(expected_size):
            for j in range(i + 1, expected_size):
                 val_ij = matrix_np[i, j]
                 val_ji = matrix_np[j, i]
                 if val_ij <= 0 or val_ji <= 0:
                     return None, f"Giá trị tại ({start_row+i+1},{start_col+j+1}) hoặc ({start_row+j+1},{start_col+i+1}) không phải là số dương."
                 if abs(val_ij * val_ji - 1.0) > 1e-4: # Increased tolerance
                    return None, f"Giá trị nghịch đảo không chính xác tại vị trí ({start_row+i+1},{start_col+j+1}) và ({start_row+j+1},{start_col+i+1}). Giá trị phải dương và A[j,i] ≈ 1/A[i,j] (tìm thấy {val_ij:.4f} và {val_ji:.4f}, tích của chúng là {val_ij*val_ji:.4f})."

        return matrix_np, None

    except Exception as e:
        traceback.print_exc()
        return None, f"Lỗi không mong muốn khi đọc file Excel: {e}"

def ahp_weighting(matrix):
    """Calculates weights, lambda_max, CI, CR using eigenvector method."""
    if matrix is None:
        return None, None, None, None, None

    n = matrix.shape[0]
    if n <= 0:
        return np.array([]), 0, 0, 0, 0

    if np.any(matrix <= 0):
         flash("Ma trận chứa giá trị không dương.", "error")
         return None, None, None, None, None
    if np.isnan(matrix).any() or np.isinf(matrix).any():
        flash("Ma trận đầu vào chứa giá trị NaN hoặc vô cực.", "error")
        return None, None, None, None, None

    try:
        eigvals, eigvecs = np.linalg.eig(matrix)
        real_eigvals = np.real(eigvals)
        lambda_max = np.max(real_eigvals)
        max_eigval_idx = np.argmax(real_eigvals)
        principal_eigvec = np.real(eigvecs[:, max_eigval_idx])
        weights = np.abs(principal_eigvec)
        sum_weights = np.sum(weights)

        if abs(sum_weights) < 1e-9:
             flash("Lỗi: Tổng vector trọng số gần bằng 0. Sử dụng trọng số bằng nhau làm dự phòng.", "warning") # Thay đổi từ error sang warning
             weights = np.ones(n) / n # Fallback
             lambda_max_for_ci = lambda_max # Vẫn dùng lambda_max tính được cho CI
        else:
             weights /= sum_weights # Normalize
             lambda_max_for_ci = lambda_max

        if n > 2:
            CI = (lambda_max_for_ci - n) / (n - 1) if (n - 1) > 1e-9 else 0.0
            CI = max(0.0, CI) # CI >= 0
            RI = RI_DICT.get(n)
            if RI is None:
                 closest_n = max([k for k in RI_DICT if k < n], default=None)
                 if closest_n:
                     RI = RI_DICT[closest_n]
                     flash(f"Cảnh báo: Không tìm thấy RI cho n={n}. Sử dụng RI cho n={closest_n} ({RI:.2f}).", "warning")
                     CR = CI / RI if RI > 1e-9 else float('inf') if CI > 1e-9 else 0.0
                 else:
                    flash(f"Lỗi: Không tìm thấy RI cho n={n} hoặc nhỏ hơn.", "error")
                    RI = None # Đặt RI là None nếu không tìm thấy
                    CR = None # CR cũng không tính được
            elif RI <= 1e-9:
                 CR = 0.0 if CI <= 1e-9 else float('inf')
            else:
                 CR = CI / RI
        else: # n <= 2
            CI = 0.0
            RI = RI_DICT.get(n, 0.00) # Lấy RI cho n=1, 2 nếu có
            CR = 0.0

        # Kiểm tra NaN/Inf trong kết quả trước khi trả về
        if any(x is not None and (math.isnan(x) or math.isinf(x)) for x in [lambda_max_for_ci, CI, CR]) or \
           np.isnan(weights).any() or np.isinf(weights).any():
            flash("Lỗi: Kết quả tính toán AHP chứa NaN hoặc vô cực.", "error")
            print(f"NaN/Inf detected: lambda_max={lambda_max_for_ci}, CI={CI}, CR={CR}, weights={weights}")
            return None, None, None, None, None # Signal error

        # Đảm bảo trọng số tổng gần bằng 1
        if abs(np.sum(weights) - 1.0) > 1e-5:
            weights /= np.sum(weights)

        return weights, lambda_max_for_ci, CI, CR, RI

    except np.linalg.LinAlgError as e:
        flash(f"Lỗi tính toán đại số tuyến tính: {e}. Ma trận có thể không hợp lệ.", "error")
        traceback.print_exc()
        return None, None, None, None, None
    except Exception as e:
        flash(f"Lỗi không mong muốn trong tính toán AHP: {e}", "error")
        traceback.print_exc()
        return None, None, None, None, None


# --- Helper Functions for Session Management ---
# (Giữ nguyên các hàm helper: clear_temporary_alt_data_for_index, clear_temporary_alt_data, clear_ahp_session_data, clear_session_data)
def clear_temporary_alt_data_for_index(index):
    """Clears temporary session keys for a specific alt comparison index."""
    keys = ['temp_alt_matrix', 'temp_alt_lambda_max', 'temp_alt_ci', 'temp_alt_cr', 'temp_alt_ri']
    for key_base in keys:
        session.pop(f'{key_base}_{index}', None)

def clear_temporary_alt_data(num_criteria):
    """Clears all temporary alt comparison keys."""
    max_crit_guess = max(num_criteria if isinstance(num_criteria, int) and num_criteria > 0 else 0, 25)
    for i in range(max_crit_guess):
        clear_temporary_alt_data_for_index(i)
    session.pop('form_data_alt', None)

def clear_ahp_session_data():
    """Clears session keys related to AHP steps (criteria onwards)."""
    keys_to_clear = [
        'selected_criteria', 'all_db_criteria', 'criteria_selected',
        'crit_matrix', 'crit_weights', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri',
        'criteria_comparison_done', 'form_data_crit',
        'alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all',
        'current_alt_criterion_index', 'alternative_comparisons_done', 'form_data_alt',
        'final_scores', 'best_alternative_info'
    ]
    num_crit_guess = len(session.get('selected_criteria', []))
    modified = False
    for key in keys_to_clear:
        if session.pop(key, None) is not None:
            modified = True
    clear_temporary_alt_data(num_crit_guess)
    if modified:
        session.modified = True

def clear_session_data():
    """Clears ALL session data related to an AHP run, including alternatives."""
    clear_ahp_session_data()
    modified = False
    if session.pop('session_alternatives', None) is not None: modified = True
    if session.pop('all_db_alternatives', None) is not None: modified = True
    if session.pop('alternatives_selected', None) is not None: modified = True
    if modified:
        session.modified = True

@app.route("/clear")
def clear_session_and_start():
    """Clears the session and redirects to the start."""
    clear_session_data()
    flash("Session đã được xóa. Bắt đầu một phân tích mới.", "info")
    return redirect(url_for('select_alternatives'))

# --- Routes ---

@app.route("/", methods=["GET"])
def index_redirect():
    """Redirects root URL to the first step."""
    return redirect(url_for('select_alternatives'))

@app.route("/select_alternatives", methods=["GET", "POST"])
def select_alternatives():
    """Step 0: User selects or enters alternatives (min MIN_ALTERNATIVES)."""
    if request.method == "POST":
        selection_mode = request.form.get('mode')
        alternatives = []
        all_db_alternatives = False

        if selection_mode == 'db':
            selected_ids_str = request.form.getlist('alternative_ids')
            if not selected_ids_str or len(selected_ids_str) < MIN_ALTERNATIVES:
                flash(f"Vui lòng chọn ít nhất {MIN_ALTERNATIVES} phương án từ cơ sở dữ liệu.", "warning")
                return redirect(url_for('select_alternatives'))
            try:
                selected_ids = [int(id_str) for id_str in selected_ids_str]
            except ValueError:
                flash("ID phương án đã chọn không hợp lệ.", "error")
                return redirect(url_for('select_alternatives'))

            format_strings = ','.join(['%s'] * len(selected_ids))
            query = f"SELECT id, ten_phuong_an FROM phuong_an WHERE id IN ({format_strings})"
            params = tuple(selected_ids)
            alternatives_from_db_unordered = execute_query(query, params, fetchall=True)

            if alternatives_from_db_unordered is None:
                 return redirect(url_for('select_alternatives'))

            if len(alternatives_from_db_unordered) != len(selected_ids):
                 flash("Không thể truy xuất tất cả phương án đã chọn hoặc ID không tồn tại.", "error")
                 return redirect(url_for('select_alternatives'))

            db_map = {item['id']: item for item in alternatives_from_db_unordered}
            alternatives = [db_map[sid] for sid in selected_ids if sid in db_map]

            if len(alternatives) != len(selected_ids):
                 flash("Lỗi sắp xếp lại phương án đã chọn.", "error")
                 return redirect(url_for('select_alternatives'))

            all_db_alternatives = True

        elif selection_mode == 'custom':
            custom_names = request.form.getlist('custom_alternative_names')
            unique_names = []
            seen_names = set()
            for name in custom_names:
                clean_name = name.strip()
                if clean_name and clean_name not in seen_names:
                    unique_names.append(clean_name)
                    seen_names.add(clean_name)

            if len(unique_names) < MIN_ALTERNATIVES:
                flash(f"Vui lòng nhập ít nhất {MIN_ALTERNATIVES} tên phương án tùy chỉnh khác nhau và không trống.", "warning")
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
    db_error = None
    query = "SELECT id, ten_phuong_an FROM phuong_an ORDER BY id"
    all_alternatives_db = execute_query(query, fetchall=True)
    if all_alternatives_db is None:
         db_error = "Lỗi lấy danh sách phương án từ DB."

    return render_template("select_alternatives.html",
                           all_alternatives_db=all_alternatives_db if all_alternatives_db else [],
                           db_error=db_error)

@app.route("/select_criteria", methods=["GET", "POST"])
def select_criteria():
    """Step 1: User selects or enters criteria (min MIN_CRITERIA)."""
    if not session.get('alternatives_selected'):
        flash("Vui lòng chọn hoặc nhập các phương án trước.", "info")
        return redirect(url_for('select_alternatives'))

    selected_alternatives = session.get('session_alternatives', [])
    if not selected_alternatives or not isinstance(selected_alternatives, list) or \
       not all(isinstance(item, dict) for item in selected_alternatives):
         flash("Dữ liệu phương án trong session không hợp lệ. Vui lòng chọn lại phương án.", "error")
         clear_session_data()
         return redirect(url_for('select_alternatives'))

    if request.method == "POST":
        selection_mode = request.form.get('mode')
        criteria = []
        all_db_criteria = False

        if selection_mode == 'db':
            selected_ids_str = request.form.getlist('criteria_ids')
            if not selected_ids_str or len(selected_ids_str) < MIN_CRITERIA:
                flash(f"Vui lòng chọn ít nhất {MIN_CRITERIA} tiêu chí từ cơ sở dữ liệu.", "warning")
                return redirect(url_for('select_criteria'))
            try:
                selected_ids = [int(id_str) for id_str in selected_ids_str]
            except ValueError:
                flash("ID tiêu chí đã chọn không hợp lệ.", "error")
                return redirect(url_for('select_criteria'))

            format_strings = ','.join(['%s'] * len(selected_ids))
            query = f"SELECT id, ten_tieu_chi FROM tieu_chi WHERE id IN ({format_strings})"
            params = tuple(selected_ids)
            criteria_from_db_unordered = execute_query(query, params, fetchall=True)

            if criteria_from_db_unordered is None:
                 return redirect(url_for('select_criteria'))

            if len(criteria_from_db_unordered) != len(selected_ids):
                 flash("Không thể truy xuất tất cả tiêu chí đã chọn hoặc ID không tồn tại.", "error")
                 return redirect(url_for('select_criteria'))

            db_map = {item['id']: item for item in criteria_from_db_unordered}
            criteria = [db_map[sid] for sid in selected_ids if sid in db_map]

            if len(criteria) != len(selected_ids):
                 flash("Lỗi sắp xếp lại tiêu chí đã chọn.", "error")
                 return redirect(url_for('select_criteria'))

            all_db_criteria = True

        elif selection_mode == 'custom':
            custom_names = request.form.getlist('custom_criteria_names')
            unique_names = []
            seen_names = set()
            for name in custom_names:
                clean_name = name.strip()
                if clean_name and clean_name not in seen_names:
                    unique_names.append(clean_name)
                    seen_names.add(clean_name)

            if len(unique_names) < MIN_CRITERIA:
                flash(f"Vui lòng nhập ít nhất {MIN_CRITERIA} tên tiêu chí tùy chỉnh khác nhau và không trống.", "warning")
                return redirect(url_for('select_criteria'))
            criteria = [{'id': None, 'ten_tieu_chi': name} for name in unique_names]
            all_db_criteria = False
        else:
            flash("Vui lòng chọn chế độ 'Sử dụng Database' hoặc 'Nhập tùy chỉnh'.", "warning")
            return redirect(url_for('select_criteria'))

        clear_ahp_session_data() # Clear only from criteria onwards
        session['selected_criteria'] = criteria
        session['all_db_criteria'] = all_db_criteria
        session['criteria_selected'] = True
        session.modified = True
        return redirect(url_for('compare_criteria'))

    # --- GET request ---
    db_error = None
    query = "SELECT id, ten_tieu_chi FROM tieu_chi ORDER BY id"
    all_criteria_db = execute_query(query, fetchall=True)
    if all_criteria_db is None:
        db_error = "Lỗi lấy danh sách tiêu chí từ DB."

    selected_alternatives_for_display = session.get('session_alternatives', [])
    return render_template("select_criteria.html",
                           all_criteria_db=all_criteria_db if all_criteria_db else [],
                           db_error=db_error,
                           selected_alternatives=selected_alternatives_for_display)

@app.route("/compare_criteria", methods=["GET", "POST"])
def compare_criteria():
    """Step 2: User compares selected criteria (manual or Excel)."""
    if not session.get('criteria_selected'):
        flash("Vui lòng chọn tiêu chí trước.", "info")
        return redirect(url_for('select_criteria'))

    selected_criteria = session.get('selected_criteria', [])
    if not selected_criteria or not isinstance(selected_criteria, list) or len(selected_criteria) < MIN_CRITERIA or \
       not all(isinstance(c, dict) and 'ten_tieu_chi' in c for c in selected_criteria):
        flash(f"Số lượng hoặc cấu trúc tiêu chí không hợp lệ. Vui lòng chọn lại.", "error")
        clear_ahp_session_data()
        return redirect(url_for('select_criteria'))

    criteria_names = [c['ten_tieu_chi'] for c in selected_criteria]
    num_criteria = len(selected_criteria)
    crit_matrix = None
    input_method = "form"

    if request.method == "POST":
        if 'criteria_excel_file' in request.files:
            file = request.files['criteria_excel_file']
            if file and file.filename != '':
                crit_matrix, error_msg = parse_excel_matrix(file, num_criteria, criteria_names)
                if error_msg:
                    flash(f"Lỗi xử lý file Excel tiêu chí: {error_msg}", "error")
                    # Clear any potentially stored calculation results from previous attempts
                    session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
                    session.pop('form_data_crit', None) # Clear form data if Excel error occurred
                    session.modified = True
                    return redirect(url_for('compare_criteria'))
                elif crit_matrix is not None:
                     input_method = "excel"
                     flash("Đã nhập ma trận so sánh tiêu chí từ file Excel.", "info")
                     session.pop('form_data_crit', None) # Clear form data if Excel is used

        if crit_matrix is None: # If Excel wasn't used or failed validation silently
            form_keys_present = any(key.startswith('pc_') for key in request.form)
            if not form_keys_present and input_method == "form":
                 flash("Vui lòng nhập giá trị so sánh thủ công hoặc tải lên file Excel.", "warning")
                 return redirect(url_for('compare_criteria'))
            # If input_method is 'excel' but crit_matrix is None here, it means parse_excel_matrix failed earlier and flashed message

            input_method = "form"
            crit_matrix = compute_pairwise_matrix("pc", criteria_names, request.form)
            if crit_matrix is None:
                session['form_data_crit'] = request.form # Preserve form data on manual input error
                # Clear calculation results
                session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
                session.modified = True
                return redirect(url_for('compare_criteria'))

        # --- Matrix obtained, proceed with AHP calculation ---
        crit_weights, crit_lambda_max, crit_ci, crit_cr, crit_ri = ahp_weighting(crit_matrix)

        if crit_weights is None:
            if input_method == "form":
                 session['form_data_crit'] = request.form # Keep form data on calc error
            # Clear potentially bad calculation results
            session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
            session.modified = True
            return redirect(url_for('compare_criteria'))

        # Store intermediate results (even if CR fails, for display)
        session['crit_matrix'] = crit_matrix.tolist()
        session['crit_lambda_max'] = crit_lambda_max
        session['crit_ci'] = crit_ci
        session['crit_cr'] = crit_cr
        session['crit_ri'] = crit_ri # Can be None
        session.modified = True # Make sure these are saved

        # CR Check
        # Allow CR to be None if RI could not be determined (ahp_weighting handles this)
        cr_check_value = crit_cr if crit_cr is not None else 0.0 # Treat None CR as acceptable for this check

        if cr_check_value > CR_THRESHOLD:
            cr_display = f"{crit_cr:.4f}" if crit_cr is not None else "Không thể tính (RI không xác định)"
            flash(f"Tỷ số nhất quán (CR = {cr_display}) vượt ngưỡng ({CR_THRESHOLD:.2f}). Vui lòng xem lại các so sánh tiêu chí.", "error")
            session['criteria_comparison_done'] = False
            if input_method == "form":
                 session['form_data_crit'] = request.form # Keep form data if manual input failed CR
            # Keep intermediate results in session for display
            session.modified = True
            return redirect(url_for('compare_criteria'))
        else:
            # --- CR is acceptable (or None) ---
            cr_display = f"{crit_cr:.4f}" if crit_cr is not None else "N/A (n<=2 hoặc RI không xác định)"
            flash(f"So sánh tiêu chí thành công (CR = {cr_display}). Tiếp tục so sánh phương án.", "success")
            session['crit_weights'] = crit_weights.tolist()
            session['criteria_comparison_done'] = True
            session.pop('form_data_crit', None) # Clear form data on success

            # Initialize structures for alternative comparisons
            num_alternatives = len(session.get('session_alternatives', []))
            if num_alternatives < MIN_ALTERNATIVES:
                 flash("Lỗi: Số lượng phương án không đủ để tiếp tục.", "error")
                 return redirect(url_for('select_alternatives'))

            session['alt_matrices_all'] = [None] * num_criteria
            session['alt_weights_all'] = [None] * num_criteria
            session['alt_lambda_max_all'] = [None] * num_criteria
            session['alt_ci_all'] = [None] * num_criteria
            session['alt_cr_all'] = [None] * num_criteria
            session['alt_ri_all'] = [None] * num_criteria
            session['current_alt_criterion_index'] = 0
            clear_temporary_alt_data(num_criteria)
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
    """Step 3: User compares alternatives for each criterion (manual or Excel)."""
    if not session.get('criteria_comparison_done'):
        flash("Vui lòng hoàn thành so sánh tiêu chí (với CR hợp lệ) trước.", "info")
        return redirect(url_for('compare_criteria'))

    # --- Retrieve and Validate Session Data ---
    selected_criteria = session.get('selected_criteria', [])
    alternatives = session.get('session_alternatives', [])

    # Validate Alternatives
    if not alternatives or not isinstance(alternatives, list) or len(alternatives) < MIN_ALTERNATIVES or \
       not all(isinstance(a, dict) and 'ten_phuong_an' in a for a in alternatives):
         flash(f"Số lượng hoặc cấu trúc phương án không hợp lệ. Vui lòng bắt đầu lại.", "error")
         clear_session_data()
         return redirect(url_for('select_alternatives'))

    # Validate Criteria
    if not selected_criteria or not isinstance(selected_criteria, list) or len(selected_criteria) < MIN_CRITERIA or \
       not all(isinstance(c, dict) and 'ten_tieu_chi' in c for c in selected_criteria):
         flash("Số lượng hoặc cấu trúc tiêu chí không hợp lệ. Vui lòng chọn lại tiêu chí.", "error")
         clear_ahp_session_data()
         return redirect(url_for('select_criteria'))

    alternative_names = [a['ten_phuong_an'] for a in alternatives]
    num_alternatives = len(alternatives)
    num_criteria = len(selected_criteria)
    current_index = session.get('current_alt_criterion_index', 0)

    # Check if comparisons are already done
    if current_index >= num_criteria:
        session['alternative_comparisons_done'] = True
        session.modified = True
        flash("Tất cả so sánh phương án đã hoàn thành.", "info")
        # Final check before results page
        alt_weights_all = session.get('alt_weights_all')
        if alt_weights_all and isinstance(alt_weights_all, list) and len(alt_weights_all) == num_criteria and \
           all(item is not None for item in alt_weights_all):
            return redirect(url_for('calculate_results'))
        else:
            flash("Dữ liệu so sánh phương án bị thiếu hoặc chưa hoàn chỉnh. Chuyển hướng về bước so sánh tiêu chí.", "warning")
            print(f"DEBUG: Redirecting from compare_alternatives (index {current_index}>={num_criteria}) due to incomplete alt_weights_all: {alt_weights_all}")
            session['current_alt_criterion_index'] = 0 # Reset index
            session.pop('alternative_comparisons_done', None)
            clear_temporary_alt_data(num_criteria) # Clear temp data
            # Re-initialize permanent storage lists for safety before redirecting back
            session['alt_matrices_all'] = [None] * num_criteria
            session['alt_weights_all'] = [None] * num_criteria
            session['alt_lambda_max_all'] = [None] * num_criteria
            session['alt_ci_all'] = [None] * num_criteria
            session['alt_cr_all'] = [None] * num_criteria
            session['alt_ri_all'] = [None] * num_criteria
            session.modified = True
            return redirect(url_for('compare_criteria'))

    current_criterion = selected_criteria[current_index]
    alt_matrix = None
    input_method = "form"

    if request.method == "POST":
         file_key = f'alternative_excel_file_{current_index}'
         if file_key in request.files:
             file = request.files[file_key]
             if file and file.filename != '':
                alt_matrix, error_msg = parse_excel_matrix(file, num_alternatives, alternative_names)
                if error_msg:
                    flash(f"Lỗi xử lý file Excel cho tiêu chí '{current_criterion['ten_tieu_chi']}': {error_msg}", "error")
                    session.pop('form_data_alt', None)
                    clear_temporary_alt_data_for_index(current_index)
                    session.modified = True
                    return redirect(url_for('compare_alternatives'))
                elif alt_matrix is not None:
                     input_method = "excel"
                     flash(f"Đã nhập ma trận so sánh phương án cho '{current_criterion['ten_tieu_chi']}' từ file Excel.", "info")
                     session.pop('form_data_alt', None) # Clear form data if excel used

         if alt_matrix is None: # If Excel not used or failed
            prefix = f"alt_pc_{current_index}"
            form_keys_present = any(key.startswith(prefix) for key in request.form)
            if not form_keys_present and input_method == "form":
                flash(f"Vui lòng nhập giá trị so sánh thủ công cho '{current_criterion['ten_tieu_chi']}' hoặc tải lên file Excel.", "warning")
                return redirect(url_for('compare_alternatives'))
            # If input_method is 'excel' but matrix is None, parse_excel_matrix failed earlier

            input_method = "form"
            alt_matrix = compute_pairwise_matrix(prefix, alternative_names, request.form)
            if alt_matrix is None:
                session['form_data_alt'] = request.form # Preserve form data
                clear_temporary_alt_data_for_index(current_index)
                session.modified = True
                return redirect(url_for('compare_alternatives'))

         # --- Matrix obtained, perform AHP calculation ---
         alt_weights, alt_lambda_max, alt_ci, alt_cr, alt_ri = ahp_weighting(alt_matrix)

         if alt_weights is None:
             if input_method == "form":
                  session['form_data_alt'] = request.form
             clear_temporary_alt_data_for_index(current_index)
             session.modified = True
             return redirect(url_for('compare_alternatives'))

         # Store results temporarily for display if CR fails
         session[f'temp_alt_matrix_{current_index}'] = alt_matrix.tolist()
         session[f'temp_alt_lambda_max_{current_index}'] = alt_lambda_max
         session[f'temp_alt_ci_{current_index}'] = alt_ci
         session[f'temp_alt_cr_{current_index}'] = alt_cr
         session[f'temp_alt_ri_{current_index}'] = alt_ri # Can be None
         session.modified = True # Save temp results

         # CR Check (Allow None CR)
         cr_check_value = alt_cr if alt_cr is not None else 0.0

         if cr_check_value > CR_THRESHOLD:
             cr_display = f"{alt_cr:.4f}" if alt_cr is not None else "Không thể tính"
             flash(f"CR cho phương án theo '{current_criterion['ten_tieu_chi']}' ({cr_display}) > {CR_THRESHOLD:.2f}. Vui lòng xem lại.", "error")
             session['alternative_comparisons_done'] = False
             if input_method == "form":
                  session['form_data_alt'] = request.form # Keep form data
             # Keep temp results, redirect back to the same criterion page
             session.modified = True
             return redirect(url_for('compare_alternatives'))
         else:
             # --- Consistent! Store results permanently ---
             cr_display = f"{alt_cr:.4f}" if alt_cr is not None else "N/A"
             flash(f"So sánh phương án theo '{current_criterion['ten_tieu_chi']}' đã lưu (CR = {cr_display}).", "success")

             def ensure_session_list(key, length, default_val=None):
                 """ Ensures a session list exists with the correct length. """
                 data = session.get(key)
                 # Check type, length, and if it contains only the default value (or expected values)
                 if not isinstance(data, list) or len(data) != length:
                    session[key] = [default_val] * length
                    print(f"DEBUG: Reinitialized session list '{key}' with length {length}")
                    session.modified = True # Mark modified when reinitializing
                 # No return needed, modifies session directly

             # Ensure permanent lists exist before assignment
             ensure_session_list('alt_matrices_all', num_criteria, default_val=None)
             ensure_session_list('alt_weights_all', num_criteria, default_val=None)
             ensure_session_list('alt_lambda_max_all', num_criteria, default_val=None)
             ensure_session_list('alt_ci_all', num_criteria, default_val=None)
             ensure_session_list('alt_cr_all', num_criteria, default_val=None)
             ensure_session_list('alt_ri_all', num_criteria, default_val=None)

             # Store permanent results (use .get to be safe, though ensure_session_list should guarantee existence)
             try:
                 session.get('alt_matrices_all', [])[current_index] = alt_matrix.tolist()
                 session.get('alt_weights_all', [])[current_index] = alt_weights.tolist()
                 session.get('alt_lambda_max_all', [])[current_index] = alt_lambda_max
                 session.get('alt_ci_all', [])[current_index] = alt_ci
                 session.get('alt_cr_all', [])[current_index] = alt_cr
                 session.get('alt_ri_all', [])[current_index] = alt_ri
                 session.modified = True # Crucial: Mark session modified after updates
             except IndexError:
                 flash(f"Lỗi nghiêm trọng: Không thể lưu kết quả vào session tại chỉ số {current_index}.", "error")
                 print(f"DEBUG: IndexError accessing session lists at index {current_index}.")
                 clear_session_data()
                 return redirect(url_for('select_alternatives'))

             # Clear temporary data for this index and general form data
             clear_temporary_alt_data_for_index(current_index)
             session.pop('form_data_alt', None) # Clear form data for this step

             # Move to the next criterion
             next_index = current_index + 1
             session['current_alt_criterion_index'] = next_index
             session.modified = True # Save the updated index

             if next_index >= num_criteria:
                 session['alternative_comparisons_done'] = True
                 session.modified = True
                 return redirect(url_for('calculate_results'))
             else:
                 # Redirect to GET for the next criterion comparison
                 return redirect(url_for('compare_alternatives'))

    # --- GET request ---
    # Retrieve temporary results for display if previous POST failed CR check
    form_data = session.get('form_data_alt', None) # Retrieve stored form data
    alt_lambda_max = session.get(f'temp_alt_lambda_max_{current_index}')
    alt_ci = session.get(f'temp_alt_ci_{current_index}')
    alt_cr = session.get(f'temp_alt_cr_{current_index}')
    alt_ri = session.get(f'temp_alt_ri_{current_index}')

    # *** Important: Clear general form data AFTER retrieving it for display
    # But DO NOT clear the temp_* results here, they are needed by the template if CR failed.
    # They will be cleared naturally when the user successfully submits for this index or moves on.
    if 'form_data_alt' in session:
         session.pop('form_data_alt')
         session.modified = True

    return render_template("compare_alternatives.html",
                           criterion=current_criterion,
                           alternatives=alternatives,
                           alternative_names=alternative_names,
                           form_data=form_data, # Pass potentially stored form data
                           alt_lambda_max=alt_lambda_max, # Pass temp results
                           alt_ci=alt_ci,
                           alt_cr=alt_cr,
                           alt_ri=alt_ri,
                           criterion_index=current_index,
                           total_criteria=num_criteria)


@app.route("/calculate_results")
def calculate_results():
    """Step 4: Calculate final scores, display results, and save to DB if applicable."""
    # --- Validation Checks ---
    if not session.get('criteria_comparison_done'):
        flash("So sánh tiêu chí chưa hoàn thành hoặc CR không hợp lệ.", "warning")
        return redirect(url_for('compare_criteria'))

    # --- Ensure alternative comparisons are marked as done ---
    num_criteria = len(session.get('selected_criteria', []))
    current_alt_index = session.get('current_alt_criterion_index', 0) # Default to 0 if key missing
    if not session.get('alternative_comparisons_done'):
        # Check if index is actually past the end
        if current_alt_index >= num_criteria:
            session['alternative_comparisons_done'] = True # Mark as done if index is validly past the end
            session.modified = True
        else:
            flash(f"So sánh phương án chưa hoàn thành (đang ở tiêu chí {current_alt_index+1}/{num_criteria}).", "warning")
            return redirect(url_for('compare_alternatives'))

    # --- Deep Validation of Required Session Data ---
    # (Giữ nguyên khối validation này, nó rất quan trọng)
    num_alternatives = len(session.get('session_alternatives', []))
    required_keys = {
        'crit_weights': (list, num_criteria), 'alt_weights_all': (list, num_criteria),
        'session_alternatives': (list, num_alternatives), 'selected_criteria': (list, num_criteria),
        'crit_matrix': (list, num_criteria), 'crit_lambda_max': ((float, int), None),
        'crit_ci': ((float, int), None), 'crit_cr': ((float, int, type(None)), None), # Allow None
        'crit_ri': ((float, int, type(None)), None), # Allow None
        'alt_matrices_all': (list, num_criteria), 'alt_lambda_max_all': (list, num_criteria),
        'alt_ci_all': (list, num_criteria), 'alt_cr_all': (list, num_criteria),
        'alt_ri_all': (list, num_criteria),
        'all_db_alternatives': (bool, None), 'all_db_criteria': (bool, None)
    }
    missing_or_invalid = []
    for key, (expected_type, expected_len) in required_keys.items():
        data = session.get(key)
        is_optional_none = key in ['crit_cr', 'crit_ri'] and data is None
        if data is None and not is_optional_none:
            missing_or_invalid.append(f"Thiếu '{key}'")
            continue
        if data is not None: # Check type only if data exists
            if isinstance(expected_type, tuple):
                if not isinstance(data, expected_type):
                    type_names = ', '.join(t.__name__ for t in expected_type)
                    missing_or_invalid.append(f"'{key}' sai kiểu (cần {type_names}, tìm thấy {type(data).__name__})")
            elif not isinstance(data, expected_type):
                missing_or_invalid.append(f"'{key}' sai kiểu (cần {expected_type.__name__}, tìm thấy {type(data).__name__})")

        if expected_len is not None and isinstance(data, list):
            if len(data) != expected_len:
                 missing_or_invalid.append(f"'{key}' sai độ dài (cần {expected_len}, tìm thấy {len(data)})")
            elif key.endswith('_all') and any(item is None for item in data): # Check for None placeholders in _all lists
                 missing_indices = [i for i, item in enumerate(data) if item is None]
                 missing_or_invalid.append(f"'{key}' thiếu dữ liệu tại chỉ số: {missing_indices}")

    if missing_or_invalid:
        error_message = "Dữ liệu session không đầy đủ/hợp lệ để tính kết quả: " + "; ".join(missing_or_invalid) + ". Vui lòng thử lại từ đầu."
        flash(error_message, "error")
        print("DEBUG: Session validation failed in calculate_results:", missing_or_invalid)
        clear_session_data()
        return redirect(url_for('select_alternatives'))

    # --- Perform Final Calculation ---
    final_scores_dict = {}
    results_display = []
    best_alternative_info = None
    calculation_error = None

    try:
        crit_weights = np.array(session['crit_weights'], dtype=float)
        alt_weights_all_list = session['alt_weights_all']
        alt_weights_matrix = np.array(alt_weights_all_list, dtype=float).T

        if crit_weights.shape != (num_criteria,): raise ValueError(f"Kích thước trọng số tiêu chí sai ({crit_weights.shape})")
        if alt_weights_matrix.shape != (num_alternatives, num_criteria): raise ValueError(f"Kích thước ma trận trọng số PA sai ({alt_weights_matrix.shape})")
        if np.isnan(crit_weights).any() or np.isinf(crit_weights).any(): raise ValueError("NaN/Inf trong trọng số tiêu chí.")
        if np.isnan(alt_weights_matrix).any() or np.isinf(alt_weights_matrix).any(): raise ValueError("NaN/Inf trong ma trận trọng số PA.")

        if abs(np.sum(crit_weights) - 1.0) > 1e-4: flash(f"Cảnh báo: Tổng trọng số tiêu chí ~ {np.sum(crit_weights):.6f}", "warning")
        col_sums = np.sum(alt_weights_matrix, axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-4):
             bad_cols = np.where(np.abs(col_sums - 1.0) > 1e-4)[0]
             flash(f"Cảnh báo: Tổng trọng số PA theo cột tiêu chí không bằng 1 (lỗi ở cột chỉ số: {bad_cols}, tổng: {col_sums[bad_cols]:.4f}).", "warning")

        final_scores_vector = np.dot(alt_weights_matrix, crit_weights)

        if final_scores_vector.shape != (num_alternatives,): raise ValueError(f"Kích thước vector điểm cuối sai ({final_scores_vector.shape})")
        if abs(np.sum(final_scores_vector) - 1.0) > 1e-4: flash(f"Cảnh báo: Tổng điểm cuối cùng ~ {np.sum(final_scores_vector):.6f}", "warning")

        alternatives_session = session['session_alternatives']
        final_scores_python = [float(score) for score in final_scores_vector] # Convert to Python floats

        if alternatives_session and len(final_scores_python) == len(alternatives_session):
            final_scores_dict = { alt['ten_phuong_an']: score for alt, score in zip(alternatives_session, final_scores_python) }
            best_alternative_name = max(final_scores_dict, key=final_scores_dict.get) if final_scores_dict else None

            for i, alt in enumerate(alternatives_session):
                alt_name = alt['ten_phuong_an']
                score = final_scores_python[i]
                is_best = (alt_name == best_alternative_name)
                display_item = {'id': alt.get('id'), 'name': alt_name, 'score': score, 'is_best': is_best}
                results_display.append(display_item)
                if is_best: best_alternative_info = display_item

            results_display.sort(key=lambda x: x['score'], reverse=True)
            session['final_scores'] = final_scores_dict
            session['best_alternative_info'] = best_alternative_info
            session.modified = True

        else:
             raise ValueError("Số lượng điểm cuối cùng không khớp với số lượng phương án trong session.")

    except (ValueError, TypeError, IndexError) as e:
         calculation_error = f"Lỗi trong quá trình tính toán cuối cùng: {e}"
         flash(calculation_error, "error")
         print(f"Final Calculation Error Details: {e}"); traceback.print_exc()
         # results_display will be empty or incomplete
    except Exception as e:
         calculation_error = f"Đã xảy ra lỗi không mong muốn trong quá trình tính toán cuối cùng: {e}"
         flash(calculation_error, "error")
         print(f"Unexpected Final Calculation Error: {e}"); traceback.print_exc()
         # results_display will be empty or incomplete


    # --- Save results to database (Only if calculation was successful) ---
    can_save_to_db = session.get('all_db_alternatives', False) and session.get('all_db_criteria', False)
    save_attempted = False
    save_successful = False

    # Chỉ thử lưu nếu không có lỗi tính toán VÀ có kết quả để hiển thị VÀ có thể lưu
    if not calculation_error and can_save_to_db and results_display:
        save_attempted = True
        conn = get_connection()
        if conn:
            analysis_group_id = str(uuid.uuid4())
            timestamp = datetime.now()
            length_mismatch_db = False # Cờ kiểm tra lỗi khớp độ dài khi lưu DB

            try:
                with conn.cursor() as cursor:
                    # --- Prepare Alternatives for DB ---
                    insert_alt_query = """
                        INSERT INTO ket_qua (analysis_group_id, thoi_gian, phuong_an_id, phuong_an_ten,
                                             tieu_chi_id, tieu_chi_ten, is_alternative, is_db_source,
                                             final_score, is_best, criterion_weight)
                        VALUES (%s, %s, %s, %s, NULL, '', TRUE, TRUE, %s, %s, NULL)
                    """
                    alt_values_to_insert = []
                    for result in results_display:
                         alt_id = result['id']
                         alt_name = result['name']
                         final_score = result['score'] # Should be float
                         is_best = result['is_best']   # Should be bool
                         if alt_id is not None: # Only save DB alternatives
                             alt_values_to_insert.append((analysis_group_id, timestamp, alt_id, alt_name, final_score, is_best))

                    # --- Prepare Criteria for DB ---
                    insert_crit_query = """
                        INSERT INTO ket_qua (analysis_group_id, thoi_gian, phuong_an_id, phuong_an_ten,
                                             tieu_chi_id, tieu_chi_ten, is_alternative, is_db_source,
                                             final_score, is_best, criterion_weight)
                        VALUES (%s, %s, NULL, '', %s, %s, FALSE, TRUE, NULL, NULL, %s)
                    """
                    crit_values_to_insert = []
                    # Sử dụng get với default là list rỗng để an toàn hơn
                    db_criteria = session.get('selected_criteria', [])
                    db_crit_weights_list = session.get('crit_weights', [])

                    # Kiểm tra khớp độ dài một cách cẩn thận
                    if len(db_criteria) == len(db_crit_weights_list):
                        for i, crit in enumerate(db_criteria):
                            crit_id = crit.get('id')
                            crit_name = crit.get('ten_tieu_chi')
                            # Đảm bảo trọng số là float
                            try:
                                crit_weight = float(db_crit_weights_list[i])
                            except (ValueError, TypeError):
                                flash(f"Lỗi kiểu dữ liệu trọng số tiêu chí '{crit_name}'. Bỏ qua lưu tiêu chí này.", "warning")
                                continue # Bỏ qua tiêu chí này nếu trọng số không hợp lệ

                            if crit_id is not None: # Chỉ lưu tiêu chí DB
                                 crit_values_to_insert.append((analysis_group_id, timestamp, crit_id, crit_name, crit_weight))
                    else:
                        length_mismatch_db = True # Đặt cờ lỗi
                        err_msg = f"Lỗi nghiêm trọng khi lưu DB: Số lượng tiêu chí ({len(db_criteria)}) và trọng số ({len(db_crit_weights_list)}) không khớp."
                        flash(err_msg, "error")
                        print(f"ERROR DB SAVE: {err_msg}")

                    # --- Execute Inserts and Commit/Rollback ---
                    # print(f"DEBUG DB SAVE: Alt values count: {len(alt_values_to_insert)}, Crit values count: {len(crit_values_to_insert)}, Length mismatch: {length_mismatch_db}")

                    db_insert_error = None # Biến để lưu lỗi insert nếu có

                    # Insert alternatives if available
                    if alt_values_to_insert:
                       try:
                           cursor.executemany(insert_alt_query, alt_values_to_insert)
                           # print("DEBUG DB SAVE: Executed alternative insert.")
                       except psycopg2.Error as alt_err:
                           db_insert_error = f"Lỗi khi chèn điểm phương án: {alt_err}"
                           print(f"ERROR DB SAVE (Alternatives): {db_insert_error}")

                    # Insert criteria if available AND no length mismatch AND no prior insert error
                    if crit_values_to_insert and not length_mismatch_db and not db_insert_error:
                        try:
                           cursor.executemany(insert_crit_query, crit_values_to_insert)
                           # print("DEBUG DB SAVE: Executed criteria insert.")
                        except psycopg2.Error as crit_err:
                            db_insert_error = f"Lỗi khi chèn trọng số tiêu chí: {crit_err}"
                            print(f"ERROR DB SAVE (Criteria): {db_insert_error}")


                    # Commit logic: Commit only if no insertion errors occurred AND no length mismatch
                    # AND both lists actually contained data (as required by can_save_to_db).
                    if not db_insert_error and not length_mismatch_db and alt_values_to_insert and crit_values_to_insert:
                       # print("DEBUG DB SAVE: Committing transaction...")
                       conn.commit()
                       save_successful = True
                       flash("Kết quả phân tích (phương án và tiêu chí) đã được lưu vào cơ sở dữ liệu.", "success")
                    else:
                       # Rollback if any error occurred or if lists were empty unexpectedly
                       # print("DEBUG DB SAVE: Rolling back transaction...")
                       conn.rollback()
                       save_successful = False
                       # Provide specific feedback for rollback reason
                       if db_insert_error:
                            flash(f"Lưu vào DB thất bại do lỗi: {db_insert_error}", "error")
                       elif length_mismatch_db:
                           flash("Lưu vào DB thất bại do lỗi không khớp giữa số lượng tiêu chí và trọng số.", "error") # Đã flash ở trên
                       elif not alt_values_to_insert:
                           flash("Lưu vào DB thất bại: Không có dữ liệu điểm phương án hợp lệ (từ DB) để lưu.", "warning")
                       elif not crit_values_to_insert:
                           # Vì can_save_to_db=True, việc này không nên xảy ra trừ khi tất cả tiêu chí DB bị lỗi get('id')
                           flash("Lưu vào DB thất bại: Không có dữ liệu trọng số tiêu chí hợp lệ (từ DB) để lưu.", "warning")
                       else:
                            flash("Lưu vào DB thất bại. Giao dịch đã được rollback.", "warning") # Fallback

            except (psycopg2.Error, Exception) as e:
                if conn: conn.rollback()
                flash(f"Lỗi trong quá trình lưu vào cơ sở dữ liệu: {e}", "error")
                print(f"DB Save Error (Outer Try): {e}"); traceback.print_exc()
                save_successful = False
            finally:
                if conn: conn.close()
        else:
            # get_connection failed
            save_attempted = True # Vẫn đánh dấu đã cố gắng
            save_successful = False # Nhưng thất bại
            # Lỗi đã được flash bởi get_connection

    elif not can_save_to_db and not calculation_error:
        flash("Kết quả được tính toán nhưng không lưu vào DB vì sử dụng phương án hoặc tiêu chí tùy chỉnh.", "info")
    elif not results_display and not calculation_error:
        flash("Không có kết quả cuối cùng để hiển thị hoặc lưu.", "warning")
    # Nếu có calculation_error, không cần flash gì thêm về việc lưu DB.

    # --- Prepare Intermediate Results for Display ---
    intermediate_results = get_intermediate_results_for_display() # Lấy dữ liệu trung gian

    return render_template("results.html",
                           results=results_display, # Sẽ rỗng nếu có lỗi tính toán
                           intermediate=intermediate_results,
                           best_alternative_info=best_alternative_info, # Sẽ None nếu có lỗi
                           save_attempted=save_attempted,
                           save_successful=save_successful,
                           can_save_to_db=can_save_to_db,
                           error=calculation_error) # Truyền lỗi tính toán vào template

# --- get_intermediate_results_for_display (Giữ nguyên) ---
def get_intermediate_results_for_display():
    """Safely retrieves intermediate results from session for the results page."""
    intermediate = {}
    try:
        crit_keys = ['selected_criteria', 'crit_matrix', 'crit_weights', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri']
        alt_keys = ['session_alternatives', 'alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all']
        for key in crit_keys + alt_keys:
            intermediate[key] = session.get(key)

        # Basic validation after retrieving all
        num_crit_check = len(intermediate.get('selected_criteria', [])) if isinstance(intermediate.get('selected_criteria'), list) else 0
        num_alt_check = len(intermediate.get('session_alternatives', [])) if isinstance(intermediate.get('session_alternatives'), list) else 0
        alt_lists_to_check = ['alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all']

        for key in alt_lists_to_check:
            data = intermediate.get(key)
            if not isinstance(data, list):
                 flash(f"Cảnh báo hiển thị: Dữ liệu trung gian '{key}' không phải là list.", "warning")
                 intermediate[key] = [None] * num_crit_check # Cố gắng tạo list rỗng
            elif len(data) != num_crit_check:
                 flash(f"Cảnh báo hiển thị: Dữ liệu trung gian '{key}' độ dài ({len(data)}) không khớp số tiêu chí ({num_crit_check}).", "warning")
                 # Pad with None if too short, or truncate if too long (less ideal but prevents crashes)
                 intermediate[key] = (data + [None] * num_crit_check)[:num_crit_check]
            elif any(item is None for item in data): # Check for None placeholders within the list
                 missing_indices = [i for i, item in enumerate(data) if item is None]
                 # Chỉ flash cảnh báo nếu nó thực sự thiếu (không phải None do lỗi CR/RI)
                 if key not in ['alt_cr_all', 'alt_ri_all']:
                      flash(f"Cảnh báo hiển thị: Dữ liệu trung gian '{key}' còn thiếu sót ở chỉ số: {missing_indices}.", "warning")

    except Exception as e:
         flash(f"Lỗi khi chuẩn bị dữ liệu trung gian để hiển thị: {e}", "warning")
         print(f"Error preparing intermediate results: {e}")
         traceback.print_exc()
         intermediate = {}
    return intermediate


@app.route("/results_history")
def results_history():
    """Displays recent results from the database, grouped by analysis run."""
    grouped_history = {}
    db_error = None

    group_query = """
        SELECT DISTINCT analysis_group_id, MAX(thoi_gian) as analysis_time
        FROM ket_qua
        WHERE is_db_source = TRUE
        GROUP BY analysis_group_id
        ORDER BY analysis_time DESC
        LIMIT 20
    """
    analysis_groups = execute_query(group_query, fetchall=True)

    if analysis_groups is None:
        db_error = "Lỗi lấy danh sách nhóm phân tích từ lịch sử."
        analysis_groups = []
    elif not analysis_groups:
        # Không có lỗi nhưng không có nhóm nào -> không cần làm gì thêm
        pass

    conn_hist = get_connection()
    if conn_hist and analysis_groups:
        try:
            with conn_hist.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                for group in analysis_groups:
                    group_id = group['analysis_group_id']
                    group_time = group['analysis_time']
                    group_data = {
                        'timestamp_obj': group_time,
                        'timestamp_str': group_time.strftime('%Y-%m-%d %H:%M:%S') if group_time else 'N/A',
                        'group_id': group_id,
                        'alternatives': [],
                        'criteria': []
                    }

                    # Get alternatives for this group
                    alt_hist_query = """
                        SELECT phuong_an_ten, final_score, is_best
                        FROM ket_qua
                        WHERE analysis_group_id = %s AND is_alternative = TRUE AND is_db_source = TRUE
                        ORDER BY final_score DESC
                    """
                    cursor.execute(alt_hist_query, (group_id,))
                    group_data['alternatives'] = [dict(row) for row in cursor.fetchall()]

                    # Get criteria for this group
                    crit_hist_query = """
                        SELECT tieu_chi_ten, criterion_weight
                        FROM ket_qua
                        WHERE analysis_group_id = %s AND is_alternative = FALSE AND is_db_source = TRUE
                        ORDER BY criterion_weight DESC
                    """
                    cursor.execute(crit_hist_query, (group_id,))
                    group_data['criteria'] = [dict(row) for row in cursor.fetchall()] # Sẽ là list rỗng nếu không tìm thấy

                    # Chỉ thêm nếu có dữ liệu phương án hoặc tiêu chí
                    if group_data['alternatives'] or group_data['criteria']:
                         grouped_history[group_id] = group_data

        except (psycopg2.Error, Exception) as e:
             db_error = f"Lỗi lấy chi tiết lịch sử: {e}"
             flash(db_error, "error"); print(f"DB History Detail Error: {e}"); traceback.print_exc()
        finally:
            if conn_hist: conn_hist.close()
    elif not conn_hist and analysis_groups:
         db_error = "Không thể kết nối DB để lấy chi tiết lịch sử."
         flash(db_error, "error")

    sorted_history_list = sorted(grouped_history.values(), key=lambda item: item['timestamp_obj'], reverse=True)

    return render_template("results_history.html", history_list=sorted_history_list, db_error=db_error)


# --- Error Handlers (Giữ nguyên) ---
@app.errorhandler(404)
def page_not_found(e):
     flash("Trang yêu cầu không được tìm thấy (404).", "error")
     return render_template('error.html', message='Trang không tìm thấy (404)'), 404

@app.errorhandler(500)
def internal_server_error(e):
     print(f"Internal Server Error: {e}")
     traceback.print_exc()
     flash("Đã xảy ra lỗi máy chủ nội bộ (500). Vui lòng thử lại sau hoặc bắt đầu lại.", "error")
     # clear_session_data() # Cân nhắc việc xóa session ở đây
     return render_template('error.html', message='Lỗi Máy chủ Nội bộ (500)'), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    max_size_mb = app.config.get('MAX_CONTENT_LENGTH', 1*1024*1024) / (1024*1024)
    flash(f"File tải lên quá lớn. Giới hạn {max_size_mb:.1f}MB.", "error")
    referer = request.headers.get("Referer")
    # Redirect back intelligently based on referer or session state
    if referer:
        if url_for('compare_alternatives') in referer: return redirect(url_for('compare_alternatives'))
        if url_for('compare_criteria') in referer: return redirect(url_for('compare_criteria'))
    if 'current_alt_criterion_index' in session: return redirect(url_for('compare_alternatives'))
    if 'criteria_selected' in session: return redirect(url_for('compare_criteria'))
    return redirect(url_for('select_alternatives'))


# --- Main Execution (Giữ nguyên) ---
if __name__ == "__main__":
    print("Kiểm tra kết nối cơ sở dữ liệu PostgreSQL...")
    conn_test = get_connection()
    if conn_test is None:
         print("\n*** CẢNH BÁO: Không thể kết nối đến cơ sở dữ liệu PostgreSQL! ***")
         print(f"1. Kiểm tra biến môi trường DATABASE_URL.")
         print("2. Kiểm tra thông tin đăng nhập/host/port/tên DB.")
         print("3. Đảm bảo PostgreSQL server đang chạy và chấp nhận kết nối.")
         print("4. Kiểm tra firewall.")
         print("5. Chạy lệnh SQL tạo bảng nếu chưa có.\n")
    else:
        print("Kết nối cơ sở dữ liệu PostgreSQL thành công.")
        try:
            with conn_test.cursor() as cursor:
                cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'phuong_an');")
                pa_exists = cursor.fetchone()[0]
                cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'tieu_chi');")
                tc_exists = cursor.fetchone()[0]
                cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'ket_qua');")
                kq_exists = cursor.fetchone()[0]
                print(f" - Bảng 'phuong_an' tồn tại: {pa_exists}")
                print(f" - Bảng 'tieu_chi' tồn tại: {tc_exists}")
                print(f" - Bảng 'ket_qua' tồn tại: {kq_exists}")
                if not (pa_exists and tc_exists and kq_exists):
                    print("*** CẢNH BÁO: Một hoặc nhiều bảng cần thiết không tồn tại trong DB! ***")
        except psycopg2.Error as db_err:
             print(f"Lỗi kiểm tra bảng trong DB: {db_err}")
        finally:
             conn_test.close()

    port = int(os.environ.get('PORT', 5001))
    is_debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    print(f"Khởi chạy ứng dụng Flask trên host 0.0.0.0, port {port} (Debug: {is_debug})...")
    app.run(debug=is_debug, host='0.0.0.0', port=port, use_reloader=is_debug)