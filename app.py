from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import psycopg2
import psycopg2.extras
from datetime import datetime
import os
import math
import traceback
import json # <--- THÊM MỚI: Cho biểu đồ và dữ liệu JSON
import pandas as pd # <--- THÊM MỚI: Cho CSV (có thể thay bằng module csv)
import csv # <--- THÊM MỚI: Một lựa chọn khác cho CSV
from werkzeug.utils import secure_filename
import uuid
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default_fallback_secret_key_for_dev_only')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 # 1 MB
# THÊM MỚI: Phần mở rộng file CSV được phép
ALLOWED_CSV_EXTENSIONS = {'csv'}

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
    return dict(
        RI_DICT=RI_DICT,
        CR_THRESHOLD=CR_THRESHOLD,
        MIN_CRITERIA=MIN_CRITERIA,
        MIN_ALTERNATIVES=MIN_ALTERNATIVES,
        json=json # <--- THÊM MỚI: Để sử dụng json.dumps trong template
    )

def get_connection():
    # ... (Giữ nguyên hàm get_connection)
    conn = None
    database_url = 'postgresql://admin:RSWYnshjkpXraGp5THCjSIaxUdtIwE4Z@dpg-d03iae2li9vc73fmjp60-a.singapore-postgres.render.com/test_2s9p'
    if not database_url:
        print("Database connection error: DATABASE_URL environment variable not set.")
        flash("Lỗi cấu hình: Không tìm thấy chuỗi kết nối cơ sở dữ liệu.", "error")
        return None
    try:
        conn = psycopg2.connect(database_url) # Use DATABASE_URL
        return conn
    except psycopg2.OperationalError as e:
        print(f"Database connection error (Operational): {e}")
        flash(f"Lỗi kết nối Database: Không thể kết nối tới server. Kiểm tra lại chuỗi kết nối và trạng thái DB.", "error")
        return None
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        flash(f"Lỗi Database: {e}", "error")
        return None
    except Exception as e:
        print(f"Unexpected error getting connection: {e}")
        flash(f"Lỗi không mong muốn khi kết nối DB: {e}", "error")
        return None


def execute_query(query, params=None, fetchone=False, fetchall=False, commit=False):
    # ... (Giữ nguyên hàm execute_query)
    conn = get_connection()
    if not conn:
        return None

    result = None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(query, params)
            if fetchone:
                result = cursor.fetchone()
                if result is not None: result = dict(result)
            elif fetchall:
                result = cursor.fetchall()
                if result is not None: result = [dict(row) for row in result]
            if commit:
                conn.commit()
    except psycopg2.Error as e:
        if conn: conn.rollback()
        print(f"Database query error: {e}\nQuery: {query}\nParams: {params}")
        traceback.print_exc()
        flash(f"Lỗi truy vấn cơ sở dữ liệu: {e}", "error")
        result = None
    except Exception as e:
        if conn: conn.rollback()
        print(f"Unexpected error during query execution: {e}")
        traceback.print_exc()
        flash(f"Lỗi không mong muốn khi thực thi truy vấn: {e}", "error")
        result = None
    finally:
        if conn: conn.close()
    return result

def compute_pairwise_matrix(prefix, item_names, form_or_dict_data):
    # THAY ĐỔI: Chấp nhận form_or_dict_data để có thể dùng cho dữ liệu từ CSV
    n = len(item_names)
    if n <= 0:
        flash("Không thể tạo ma trận so sánh với 0 phần tử.", "error")
        return None
    matrix = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            key = f"{prefix}_{i}_{j}"
            # THAY ĐỔI: Lấy giá trị từ form hoặc dictionary
            if isinstance(form_or_dict_data, dict):
                val_str = form_or_dict_data.get(key)
            else: # Giả sử là request.form
                val_str = form_or_dict_data.get(key)


            if val_str is None or str(val_str).strip() == "": # Chuyển val_str sang string để strip
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
    # ... (Giữ nguyên hàm parse_excel_matrix)
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
    # ... (Giữ nguyên hàm ahp_weighting)
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
             flash("Lỗi: Tổng vector trọng số gần bằng 0. Sử dụng trọng số bằng nhau làm dự phòng.", "warning")
             weights = np.ones(n) / n
             lambda_max_for_ci = lambda_max
        else:
             weights /= sum_weights
             lambda_max_for_ci = lambda_max

        if n > 2:
            CI = (lambda_max_for_ci - n) / (n - 1) if (n - 1) > 1e-9 else 0.0
            CI = max(0.0, CI)
            RI = RI_DICT.get(n)
            if RI is None:
                 closest_n = max([k for k in RI_DICT if k < n], default=None)
                 if closest_n:
                     RI = RI_DICT[closest_n]
                     flash(f"Cảnh báo: Không tìm thấy RI cho n={n}. Sử dụng RI cho n={closest_n} ({RI:.2f}).", "warning")
                     CR = CI / RI if RI > 1e-9 else float('inf') if CI > 1e-9 else 0.0
                 else:
                    flash(f"Lỗi: Không tìm thấy RI cho n={n} hoặc nhỏ hơn.", "error")
                    RI = None
                    CR = None
            elif RI <= 1e-9:
                 CR = 0.0 if CI <= 1e-9 else float('inf')
            else:
                 CR = CI / RI
        else: # n <= 2
            CI = 0.0
            RI = RI_DICT.get(n, 0.00)
            CR = 0.0

        if any(x is not None and (math.isnan(x) or math.isinf(x)) for x in [lambda_max_for_ci, CI, CR]) or \
           np.isnan(weights).any() or np.isinf(weights).any():
            flash("Lỗi: Kết quả tính toán AHP chứa NaN hoặc vô cực.", "error")
            print(f"NaN/Inf detected: lambda_max={lambda_max_for_ci}, CI={CI}, CR={CR}, weights={weights}")
            return None, None, None, None, None

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
def clear_temporary_alt_data_for_index(index):
    keys = ['temp_alt_matrix', 'temp_alt_lambda_max', 'temp_alt_ci', 'temp_alt_cr', 'temp_alt_ri']
    for key_base in keys:
        session.pop(f'{key_base}_{index}', None)

def clear_temporary_alt_data(num_criteria):
    max_crit_guess = max(num_criteria if isinstance(num_criteria, int) and num_criteria > 0 else 0, 25)
    for i in range(max_crit_guess):
        clear_temporary_alt_data_for_index(i)
    session.pop('form_data_alt', None)

def clear_ahp_session_data():
    keys_to_clear = [
        'selected_criteria', 'all_db_criteria', 'criteria_selected',
        'crit_matrix', 'crit_weights', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri',
        'criteria_comparison_done', 'form_data_crit',
        'alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all',
        'current_alt_criterion_index', 'alternative_comparisons_done', 'form_data_alt',
        'final_scores', 'best_alternative_info',
        # THÊM MỚI: Dọn dẹp session liên quan đến CSV
        'csv_preview_data', 'csv_parsed_raw_data', 'csv_filename', 'csv_input_error'
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
    clear_ahp_session_data()
    modified = False
    if session.pop('session_alternatives', None) is not None: modified = True
    if session.pop('all_db_alternatives', None) is not None: modified = True
    if session.pop('alternatives_selected', None) is not None: modified = True
    if modified:
        session.modified = True

@app.route("/clear")
def clear_session_and_start():
    clear_session_data()
    flash("Session đã được xóa. Bắt đầu một phân tích mới.", "info")
    return redirect(url_for('select_alternatives'))

# --- Routes ---

@app.route("/", methods=["GET"])
def index_redirect():
    return redirect(url_for('select_alternatives'))

@app.route("/select_alternatives", methods=["GET", "POST"])
def select_alternatives():
    # ... (Logic hiện tại của select_alternatives, không thay đổi nhiều ở đây)
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
    # clear_session_data() # Không nên clear ở đây nữa nếu người dùng đang quay lại từ bước sau
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
    # ... (Logic hiện tại của select_criteria, không thay đổi nhiều)
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

        clear_ahp_session_data()
        session['selected_criteria'] = criteria
        session['all_db_criteria'] = all_db_criteria
        session['criteria_selected'] = True
        session.modified = True
        return redirect(url_for('compare_criteria'))

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


# --- THÊM MỚI: Route để thêm phương án/tiêu chí vào DB ---
@app.route("/add_item_to_db", methods=["POST"])
def add_item_to_db():
    item_type = request.form.get('item_type')
    new_item_name = request.form.get('new_item_name', '').strip()
    redirect_url_name = 'select_alternatives' # Default redirect

    if not new_item_name:
        flash("Tên không được để trống.", "error")
    elif item_type == 'alternative':
        table_name = 'phuong_an'
        column_name = 'ten_phuong_an'
        redirect_url_name = 'select_alternatives'
    elif item_type == 'criterion':
        table_name = 'tieu_chi'
        column_name = 'ten_tieu_chi'
        redirect_url_name = 'select_criteria'
    else:
        flash("Loại mục không hợp lệ.", "error")
        return redirect(url_for(redirect_url_name))

    if new_item_name:
        # Kiểm tra trùng lặp
        check_query = f"SELECT id FROM {table_name} WHERE {column_name} = %s"
        existing_item = execute_query(check_query, (new_item_name,), fetchone=True)
        if existing_item:
            flash(f"Tên '{new_item_name}' đã tồn tại trong cơ sở dữ liệu.", "warning")
        else:
            insert_query = f"INSERT INTO {table_name} ({column_name}) VALUES (%s)"
            # execute_query đã có commit=True
            success = execute_query(insert_query, (new_item_name,), commit=True)
            # Kiểm tra thành công dựa trên việc có lỗi hay không (execute_query sẽ flash lỗi)
            # và có thể kiểm tra lại bằng cách select
            check_again = execute_query(check_query, (new_item_name,), fetchone=True)
            if check_again:
                flash(f"Đã thêm '{new_item_name}' vào cơ sở dữ liệu thành công!", "success")
            else:
                # Lỗi có thể đã được flash bởi execute_query, hoặc có thể là lỗi logic khác
                flash(f"Không thể thêm '{new_item_name}' vào cơ sở dữ liệu. Kiểm tra log.", "error")

    return redirect(url_for(redirect_url_name))


@app.route("/compare_criteria", methods=["GET", "POST"])
def compare_criteria():
    # ... (Logic hiện tại, không thay đổi nhiều về cốt lõi)
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
                    session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
                    session.pop('form_data_crit', None)
                    session.modified = True
                    return redirect(url_for('compare_criteria'))
                elif crit_matrix is not None:
                     input_method = "excel"
                     flash("Đã nhập ma trận so sánh tiêu chí từ file Excel.", "info")
                     session.pop('form_data_crit', None)

        if crit_matrix is None:
            form_keys_present = any(key.startswith('pc_') for key in request.form)
            if not form_keys_present and input_method == "form":
                 flash("Vui lòng nhập giá trị so sánh thủ công hoặc tải lên file Excel.", "warning")
                 return redirect(url_for('compare_criteria'))
            input_method = "form"
            crit_matrix = compute_pairwise_matrix("pc", criteria_names, request.form)
            if crit_matrix is None:
                session['form_data_crit'] = request.form
                session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
                session.modified = True
                return redirect(url_for('compare_criteria'))

        crit_weights, crit_lambda_max, crit_ci, crit_cr, crit_ri = ahp_weighting(crit_matrix)

        if crit_weights is None:
            if input_method == "form":
                 session['form_data_crit'] = request.form
            session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
            session.modified = True
            return redirect(url_for('compare_criteria'))

        session['crit_matrix'] = crit_matrix.tolist()
        session['crit_lambda_max'] = crit_lambda_max
        session['crit_ci'] = crit_ci
        session['crit_cr'] = crit_cr
        session['crit_ri'] = crit_ri
        session.modified = True

        cr_check_value = crit_cr if crit_cr is not None else 0.0
        if cr_check_value > CR_THRESHOLD:
            cr_display = f"{crit_cr:.4f}" if crit_cr is not None else "Không thể tính (RI không xác định)"
            flash(f"Tỷ số nhất quán (CR = {cr_display}) vượt ngưỡng ({CR_THRESHOLD:.2f}). Vui lòng xem lại các so sánh tiêu chí.", "error")
            session['criteria_comparison_done'] = False
            if input_method == "form":
                 session['form_data_crit'] = request.form
            session.modified = True
            return redirect(url_for('compare_criteria'))
        else:
            cr_display = f"{crit_cr:.4f}" if crit_cr is not None else "N/A (n<=2 hoặc RI không xác định)"
            flash(f"So sánh tiêu chí thành công (CR = {cr_display}). Tiếp tục so sánh phương án.", "success")
            session['crit_weights'] = crit_weights.tolist()
            session['criteria_comparison_done'] = True
            session.pop('form_data_crit', None)

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
    # ... (Logic hiện tại, không thay đổi nhiều về cốt lõi)
    if not session.get('criteria_comparison_done'):
        flash("Vui lòng hoàn thành so sánh tiêu chí (với CR hợp lệ) trước.", "info")
        return redirect(url_for('compare_criteria'))

    selected_criteria = session.get('selected_criteria', [])
    alternatives = session.get('session_alternatives', [])

    if not alternatives or not isinstance(alternatives, list) or len(alternatives) < MIN_ALTERNATIVES or \
       not all(isinstance(a, dict) and 'ten_phuong_an' in a for a in alternatives):
         flash(f"Số lượng hoặc cấu trúc phương án không hợp lệ. Vui lòng bắt đầu lại.", "error")
         clear_session_data()
         return redirect(url_for('select_alternatives'))
    if not selected_criteria or not isinstance(selected_criteria, list) or len(selected_criteria) < MIN_CRITERIA or \
       not all(isinstance(c, dict) and 'ten_tieu_chi' in c for c in selected_criteria):
         flash("Số lượng hoặc cấu trúc tiêu chí không hợp lệ. Vui lòng chọn lại tiêu chí.", "error")
         clear_ahp_session_data()
         return redirect(url_for('select_criteria'))

    alternative_names = [a['ten_phuong_an'] for a in alternatives]
    num_alternatives = len(alternatives)
    num_criteria = len(selected_criteria)
    current_index = session.get('current_alt_criterion_index', 0)

    if current_index >= num_criteria:
        session['alternative_comparisons_done'] = True
        session.modified = True
        flash("Tất cả so sánh phương án đã hoàn thành.", "info")
        alt_weights_all = session.get('alt_weights_all')
        if alt_weights_all and isinstance(alt_weights_all, list) and len(alt_weights_all) == num_criteria and \
           all(item is not None for item in alt_weights_all):
            return redirect(url_for('calculate_results'))
        else:
            flash("Dữ liệu so sánh phương án bị thiếu hoặc chưa hoàn chỉnh. Chuyển hướng về bước so sánh tiêu chí.", "warning")
            session['current_alt_criterion_index'] = 0
            session.pop('alternative_comparisons_done', None)
            clear_temporary_alt_data(num_criteria)
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
                     session.pop('form_data_alt', None)

         if alt_matrix is None:
            prefix = f"alt_pc_{current_index}"
            form_keys_present = any(key.startswith(prefix) for key in request.form)
            if not form_keys_present and input_method == "form":
                flash(f"Vui lòng nhập giá trị so sánh thủ công cho '{current_criterion['ten_tieu_chi']}' hoặc tải lên file Excel.", "warning")
                return redirect(url_for('compare_alternatives'))
            input_method = "form"
            alt_matrix = compute_pairwise_matrix(prefix, alternative_names, request.form)
            if alt_matrix is None:
                session['form_data_alt'] = request.form
                clear_temporary_alt_data_for_index(current_index)
                session.modified = True
                return redirect(url_for('compare_alternatives'))

         alt_weights, alt_lambda_max, alt_ci, alt_cr, alt_ri = ahp_weighting(alt_matrix)

         if alt_weights is None:
             if input_method == "form":
                  session['form_data_alt'] = request.form
             clear_temporary_alt_data_for_index(current_index)
             session.modified = True
             return redirect(url_for('compare_alternatives'))

         session[f'temp_alt_matrix_{current_index}'] = alt_matrix.tolist()
         session[f'temp_alt_lambda_max_{current_index}'] = alt_lambda_max
         session[f'temp_alt_ci_{current_index}'] = alt_ci
         session[f'temp_alt_cr_{current_index}'] = alt_cr
         session[f'temp_alt_ri_{current_index}'] = alt_ri
         session.modified = True

         cr_check_value = alt_cr if alt_cr is not None else 0.0
         if cr_check_value > CR_THRESHOLD:
             cr_display = f"{alt_cr:.4f}" if alt_cr is not None else "Không thể tính"
             flash(f"CR cho phương án theo '{current_criterion['ten_tieu_chi']}' ({cr_display}) > {CR_THRESHOLD:.2f}. Vui lòng xem lại.", "error")
             session['alternative_comparisons_done'] = False
             if input_method == "form":
                  session['form_data_alt'] = request.form
             session.modified = True
             return redirect(url_for('compare_alternatives'))
         else:
             cr_display = f"{alt_cr:.4f}" if alt_cr is not None else "N/A"
             flash(f"So sánh phương án theo '{current_criterion['ten_tieu_chi']}' đã lưu (CR = {cr_display}).", "success")
             def ensure_session_list(key, length, default_val=None):
                 data = session.get(key)
                 if not isinstance(data, list) or len(data) != length:
                    session[key] = [default_val] * length
                    session.modified = True
             ensure_session_list('alt_matrices_all', num_criteria, default_val=None)
             ensure_session_list('alt_weights_all', num_criteria, default_val=None)
             ensure_session_list('alt_lambda_max_all', num_criteria, default_val=None)
             ensure_session_list('alt_ci_all', num_criteria, default_val=None)
             ensure_session_list('alt_cr_all', num_criteria, default_val=None)
             ensure_session_list('alt_ri_all', num_criteria, default_val=None)
             try:
                 session.get('alt_matrices_all', [])[current_index] = alt_matrix.tolist()
                 session.get('alt_weights_all', [])[current_index] = alt_weights.tolist()
                 session.get('alt_lambda_max_all', [])[current_index] = alt_lambda_max
                 session.get('alt_ci_all', [])[current_index] = alt_ci
                 session.get('alt_cr_all', [])[current_index] = alt_cr
                 session.get('alt_ri_all', [])[current_index] = alt_ri
                 session.modified = True
             except IndexError:
                 flash(f"Lỗi nghiêm trọng: Không thể lưu kết quả vào session tại chỉ số {current_index}.", "error")
                 clear_session_data()
                 return redirect(url_for('select_alternatives'))

             clear_temporary_alt_data_for_index(current_index)
             session.pop('form_data_alt', None)
             next_index = current_index + 1
             session['current_alt_criterion_index'] = next_index
             session.modified = True
             if next_index >= num_criteria:
                 session['alternative_comparisons_done'] = True
                 session.modified = True
                 return redirect(url_for('calculate_results'))
             else:
                 return redirect(url_for('compare_alternatives'))

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

# --- THÊM MỚI: Các hàm và route cho xử lý CSV ---
def allowed_csv_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS

def parse_single_csv_ahp_data(file_storage):
    """
    Phân tích một file CSV duy nhất chứa tất cả dữ liệu AHP.
    Trả về thêm dữ liệu để hiển thị ma trận preview.
    """
    if not file_storage or file_storage.filename == '':
        return None, "Không có file nào được chọn.", None
    if not allowed_csv_file(file_storage.filename):
        return None, "Định dạng file không hợp lệ. Chỉ chấp nhận file .csv", None

    try:
        file_storage.seek(0)
        stream = file_storage.stream.read().decode("utf-8")
        csv_reader = csv.reader(stream.splitlines())
        header = next(csv_reader, None)
        if header and header[0].strip().upper() == 'TYPE':
            pass
        else:
            file_storage.seek(0)
            stream = file_storage.stream.read().decode("utf-8")
            csv_reader = csv.reader(stream.splitlines())
    except Exception as e:
        traceback.print_exc()
        return None, f"Lỗi khi đọc file CSV: {e}", None

    criteria_names = []
    alternatives_names = []
    
    # Dữ liệu cho việc build matrix trong backend
    criteria_comparisons_for_backend = {} 
    alternatives_comparisons_for_backend = {} 
    
    # Dữ liệu cho việc hiển thị ma trận preview trong frontend
    criteria_matrix_preview = {} # Sẽ có dạng {("ItemA", "ItemB"): value}
    alternatives_matrices_preview = {} # Sẽ có dạng {"CriterionCtx": {("AltX", "AltY"): value}}

    parsed_rows_for_preview_raw = [] # Vẫn giữ lại để debug hoặc hiển thị raw nếu cần

    temp_crit_names = set()
    temp_alt_names = set()

    file_storage.seek(0)
    stream_for_names = file_storage.stream.read().decode("utf-8")
    name_reader = csv.reader(stream_for_names.splitlines())
    if header and header[0].strip().upper() == 'TYPE': next(name_reader, None)

    for row_idx, row in enumerate(name_reader):
        if len(row) < 4: continue
        row_type = row[0].strip().upper()
        item_row = row[1].strip()
        item_col = row[2].strip()
        if not item_row or not item_col: continue

        if row_type == "CRITERIA":
            temp_crit_names.add(item_row)
            temp_crit_names.add(item_col)
        elif row_type == "ALTERNATIVES":
            temp_alt_names.add(item_row)
            temp_alt_names.add(item_col)
            if len(row) >= 5 and row[4].strip():
                temp_crit_names.add(row[4].strip())
    
    criteria_names = sorted(list(temp_crit_names))
    alternatives_names = sorted(list(temp_alt_names))

    if not criteria_names or len(criteria_names) < MIN_CRITERIA:
        return None, f"Không đủ tên tiêu chí (cần {MIN_CRITERIA}, tìm thấy {len(criteria_names)}).", None
    if not alternatives_names or len(alternatives_names) < MIN_ALTERNATIVES:
        return None, f"Không đủ tên phương án (cần {MIN_ALTERNATIVES}, tìm thấy {len(alternatives_names)}).", None

    # Build name-to-index maps for easier lookup
    crit_name_to_idx = {name: i for i, name in enumerate(criteria_names)}
    alt_name_to_idx = {name: i for i, name in enumerate(alternatives_names)}

    file_storage.seek(0)
    stream_for_values = file_storage.stream.read().decode("utf-8")
    value_reader = csv.reader(stream_for_values.splitlines())
    if header and header[0].strip().upper() == 'TYPE': next(value_reader, None)

    for row_idx, row in enumerate(value_reader):
        parsed_rows_for_preview_raw.append(list(row))
        if len(row) < 4: continue
        
        row_type = row[0].strip().upper()
        item_row_name = row[1].strip()
        item_col_name = row[2].strip()
        value_str = row[3].strip()

        if not item_row_name or not item_col_name or not value_str: continue
        
        try:
            value = float(eval(value_str))
            if value <= 0:
                return None, f"Giá trị ở dòng {row_idx+2} ('{value_str}') phải dương.", None
        except:
            return None, f"Giá trị ở dòng {row_idx+2} ('{value_str}') không hợp lệ.", None

        if row_type == "CRITERIA":
            if item_row_name not in crit_name_to_idx or item_col_name not in crit_name_to_idx:
                return None, f"Tên TC không hợp lệ ở dòng {row_idx+2}.", None
            
            idx_row = crit_name_to_idx[item_row_name]
            idx_col = crit_name_to_idx[item_col_name]

            if idx_row < idx_col:
                criteria_comparisons_for_backend[f"pc_{idx_row}_{idx_col}"] = value
                criteria_matrix_preview[(item_row_name, item_col_name)] = value
            elif idx_col < idx_row:
                criteria_comparisons_for_backend[f"pc_{idx_col}_{idx_row}"] = 1.0/value
                criteria_matrix_preview[(item_col_name, item_row_name)] = 1.0/value


        elif row_type == "ALTERNATIVES":
            if len(row) < 5 or not row[4].strip():
                return None, f"Dòng {row_idx+2} (ALTERNATIVES) thiếu context tiêu chí.", None
            
            criterion_context = row[4].strip()
            if criterion_context not in crit_name_to_idx:
                return None, f"Context tiêu chí '{criterion_context}' (dòng {row_idx+2}) không hợp lệ.", None
            if item_row_name not in alt_name_to_idx or item_col_name not in alt_name_to_idx:
                 return None, f"Tên PA không hợp lệ (dòng {row_idx+2}, context {criterion_context}).", None

            if criterion_context not in alternatives_matrices_preview:
                alternatives_matrices_preview[criterion_context] = {}
            if criterion_context not in alternatives_comparisons_for_backend: # Thay đổi key
                alternatives_comparisons_for_backend[criterion_context] = {}

            criterion_idx_for_prefix = crit_name_to_idx[criterion_context]
            idx_row_alt = alt_name_to_idx[item_row_name]
            idx_col_alt = alt_name_to_idx[item_col_name]
            
            # Dùng cho backend processing
            if idx_row_alt < idx_col_alt:
                key_backend = f"alt_pc_{criterion_idx_for_prefix}_{idx_row_alt}_{idx_col_alt}"
                alternatives_comparisons_for_backend[criterion_context][key_backend] = value
                alternatives_matrices_preview[criterion_context][(item_row_name, item_col_name)] = value
            elif idx_col_alt < idx_row_alt:
                key_backend = f"alt_pc_{criterion_idx_for_prefix}_{idx_col_alt}_{idx_row_alt}"
                alternatives_comparisons_for_backend[criterion_context][key_backend] = 1.0/value
                alternatives_matrices_preview[criterion_context][(item_col_name, item_row_name)] = 1.0/value

    # Kiểm tra số lượng so sánh (giữ nguyên logic kiểm tra backend data)
    num_crit = len(criteria_names)
    expected_crit_comps = (num_crit * (num_crit - 1)) / 2
    if len(criteria_comparisons_for_backend) < expected_crit_comps:
        return None, f"Không đủ so sánh cho TC (cần {int(expected_crit_comps)}, có {len(criteria_comparisons_for_backend)}).", None

    num_alt = len(alternatives_names)
    expected_alt_comps = (num_alt * (num_alt - 1)) / 2
    for crit_ctx, alt_comps_dict in alternatives_comparisons_for_backend.items():
        if len(alt_comps_dict) < expected_alt_comps:
            return None, f"Không đủ so sánh PA cho TC '{crit_ctx}' (cần {int(expected_alt_comps)}, có {len(alt_comps_dict)}).", None
    if len(alternatives_comparisons_for_backend) < num_crit:
         return None, f"Thiếu dữ liệu so sánh PA cho một số TC (cần {num_crit} bộ, có {len(alternatives_comparisons_for_backend)}).", None
    
    # Chuẩn bị dữ liệu đầy đủ hơn cho preview template
    # Chúng ta cần các ma trận hoàn chỉnh (bao gồm cả giá trị nghịch đảo và đường chéo)
    # để template có thể render.
    
    # 1. Criteria Preview Matrix (full)
    crit_preview_full_matrix = np.ones((num_crit, num_crit), dtype=object) # Dùng object để chứa số hoặc '-'
    for i in range(num_crit):
        for j in range(num_crit):
            name_i = criteria_names[i]
            name_j = criteria_names[j]
            if i == j:
                crit_preview_full_matrix[i, j] = 1.0
            elif i < j:
                val = criteria_matrix_preview.get((name_i, name_j))
                crit_preview_full_matrix[i, j] = round(val, 4) if val is not None else "Lỗi"
                crit_preview_full_matrix[j, i] = round(1.0/val, 4) if val is not None and val != 0 else "Lỗi"
            # elif j < i: (đã xử lý bởi i < j)
            #     pass

    # 2. Alternatives Preview Matrices (full)
    alt_previews_full_matrices = {}
    for crit_name_ctx, comparisons in alternatives_matrices_preview.items():
        alt_matrix_for_crit = np.ones((num_alt, num_alt), dtype=object)
        for i in range(num_alt):
            for j in range(num_alt):
                alt_name_i = alternatives_names[i]
                alt_name_j = alternatives_names[j]
                if i == j:
                    alt_matrix_for_crit[i, j] = 1.0
                elif i < j:
                    val = comparisons.get((alt_name_i, alt_name_j))
                    alt_matrix_for_crit[i, j] = round(val, 4) if val is not None else "Lỗi"
                    alt_matrix_for_crit[j, i] = round(1.0/val, 4) if val is not None and val != 0 else "Lỗi"
                # elif j < i:
                #     pass
        alt_previews_full_matrices[crit_name_ctx] = alt_matrix_for_crit.tolist()


    return {
        "criteria_names": criteria_names,
        "alternatives_names": alternatives_names,
        "criteria_comparisons_dict": criteria_comparisons_for_backend, # Cho backend
        "alternatives_comparisons_dict": alternatives_comparisons_for_backend, # Cho backend
        "parsed_rows_for_preview_raw": parsed_rows_for_preview_raw, # Vẫn giữ lại
        # THÊM MỚI cho preview dạng ma trận
        "criteria_preview_matrix_render": crit_preview_full_matrix.tolist(),
        "alternatives_preview_matrices_render": alt_previews_full_matrices
    }, None, parsed_rows_for_preview_raw # Trả về thêm raw_rows để route sử dụng


@app.route("/upload_csv", methods=["GET", "POST"])
def upload_csv():
    if request.method == "POST":
        # --- Xử lý nút "Xác nhận và Tính toán" ---
        if 'confirm_csv_data' in request.form:
            csv_preview_data = session.get('csv_preview_data')
            if not csv_preview_data:
                flash("Không có dữ liệu CSV để xác nhận. Vui lòng tải lên lại.", "error")
                return redirect(url_for('upload_csv'))

            criteria_names = csv_preview_data.get('criteria_names')
            alternatives_names = csv_preview_data.get('alternatives_names')
            # Sử dụng dict đã được chuẩn bị cho backend từ hàm parse
            crit_comp_dict_for_backend = csv_preview_data.get('criteria_comparisons_dict')
            alt_comp_dict_by_crit_name_for_backend = csv_preview_data.get('alternatives_comparisons_dict')

            # Kiểm tra dữ liệu cơ bản từ session
            if not all([criteria_names, alternatives_names, crit_comp_dict_for_backend, alt_comp_dict_by_crit_name_for_backend]):
                flash("Dữ liệu CSV trong session không đầy đủ để xác nhận. Vui lòng tải lại file.", "error")
                session['csv_input_error'] = True
                return redirect(url_for('upload_csv'))


            # --- Xóa session cũ và thiết lập AHP từ dữ liệu CSV ---
            clear_session_data() # Xóa toàn bộ session AHP trước đó

            # 1. Thiết lập alternatives và criteria
            session['session_alternatives'] = [{'id': None, 'ten_phuong_an': name} for name in alternatives_names]
            session['selected_criteria'] = [{'id': None, 'ten_tieu_chi': name} for name in criteria_names]
            session['all_db_alternatives'] = False # Vì từ CSV
            session['all_db_criteria'] = False   # Vì từ CSV
            session['alternatives_selected'] = True
            session['criteria_selected'] = True

            num_criteria = len(criteria_names)
            num_alternatives = len(alternatives_names)

            # 2. Xử lý ma trận tiêu chí
            crit_matrix = compute_pairwise_matrix("pc", criteria_names, crit_comp_dict_for_backend)
            if crit_matrix is None:
                # Lỗi đã được flash bởi compute_pairwise_matrix
                session['csv_input_error'] = True # Đánh dấu có lỗi để hiển thị lại preview (nếu cần)
                return redirect(url_for('upload_csv'))

            crit_weights, crit_lambda_max, crit_ci, crit_cr, crit_ri = ahp_weighting(crit_matrix)
            if crit_weights is None:
                # Lỗi đã được flash bởi ahp_weighting
                session['csv_input_error'] = True
                return redirect(url_for('upload_csv'))

            if crit_cr is not None and crit_cr > CR_THRESHOLD:
                flash(f"CR cho ma trận tiêu chí ({crit_cr:.4f}) vượt ngưỡng ({CR_THRESHOLD:.2f}). Vui lòng kiểm tra lại file CSV.", "error")
                session['csv_input_error'] = True
                # Không lưu kết quả lỗi vào session vĩnh viễn
                return redirect(url_for('upload_csv'))

            session['crit_matrix'] = crit_matrix.tolist()
            session['crit_weights'] = crit_weights.tolist()
            session['crit_lambda_max'] = crit_lambda_max
            session['crit_ci'] = crit_ci
            session['crit_cr'] = crit_cr
            session['crit_ri'] = crit_ri
            session['criteria_comparison_done'] = True

            # 3. Xử lý ma trận phương án
            session['alt_matrices_all'] = [None] * num_criteria
            session['alt_weights_all'] = [None] * num_criteria
            session['alt_lambda_max_all'] = [None] * num_criteria
            session['alt_ci_all'] = [None] * num_criteria
            session['alt_cr_all'] = [None] * num_criteria
            session['alt_ri_all'] = [None] * num_criteria

            # Lặp qua tên tiêu chí đã được sắp xếp từ hàm parse
            for i, crit_name_context in enumerate(criteria_names):
                # alt_comp_dict_by_crit_name_for_backend có key là tên tiêu chí
                alt_comp_data_for_this_crit = alt_comp_dict_by_crit_name_for_backend.get(crit_name_context)
                
                if not alt_comp_data_for_this_crit:
                    flash(f"Thiếu dữ liệu so sánh phương án cho tiêu chí '{crit_name_context}' trong file CSV đã xử lý.", "error")
                    session['csv_input_error'] = True
                    return redirect(url_for('upload_csv'))

                # compute_pairwise_matrix yêu cầu dict có key dạng "alt_pc_{crit_idx}_i_j"
                # Dữ liệu trong alt_comp_data_for_this_crit đã có key đúng dạng này
                alt_matrix = compute_pairwise_matrix(f"alt_pc_{i}", alternatives_names, alt_comp_data_for_this_crit)

                if alt_matrix is None:
                    flash(f"Lỗi tạo ma trận phương án cho tiêu chí '{crit_name_context}'.", "error")
                    session['csv_input_error'] = True
                    return redirect(url_for('upload_csv'))

                alt_weights, alt_lambda_max, alt_ci, alt_cr, alt_ri = ahp_weighting(alt_matrix)
                if alt_weights is None:
                    flash(f"Lỗi tính toán AHP cho phương án theo tiêu chí '{crit_name_context}'.", "error")
                    session['csv_input_error'] = True
                    return redirect(url_for('upload_csv'))

                if alt_cr is not None and alt_cr > CR_THRESHOLD:
                    flash(f"CR cho phương án theo tiêu chí '{crit_name_context}' ({alt_cr:.4f}) vượt ngưỡng ({CR_THRESHOLD:.2f}). Kiểm tra lại file CSV.", "error")
                    session['csv_input_error'] = True
                    return redirect(url_for('upload_csv'))

                # Lưu kết quả vào session['alt_..._all'] tại đúng chỉ số i
                session['alt_matrices_all'][i] = alt_matrix.tolist()
                session['alt_weights_all'][i] = alt_weights.tolist()
                session['alt_lambda_max_all'][i] = alt_lambda_max
                session['alt_ci_all'][i] = alt_ci
                session['alt_cr_all'][i] = alt_cr
                session['alt_ri_all'][i] = alt_ri

            session['alternative_comparisons_done'] = True
            session['current_alt_criterion_index'] = num_criteria # Đánh dấu đã xong tất cả so sánh PA

            # Xóa dữ liệu CSV preview khỏi session sau khi xử lý thành công
            session.pop('csv_preview_data', None)
            session.pop('csv_filename', None)
            session.pop('csv_input_error', None) # Quan trọng: xóa cờ lỗi
            session.modified = True

            flash("Dữ liệu từ file CSV đã được xử lý thành công. Xem kết quả.", "success")
            return redirect(url_for('calculate_results'))

        # --- Xử lý tải file lên để xem trước ---
        if 'csv_file' not in request.files:
            flash('Không có phần file nào trong request.', "error")
            return redirect(request.url) # Redirect lại chính trang upload_csv
        
        file = request.files['csv_file']
        if file.filename == '':
            flash('Chưa chọn file nào.', "warning")
            return redirect(request.url)

        if file and allowed_csv_file(file.filename):
            filename = secure_filename(file.filename)
            # parse_single_csv_ahp_data giờ trả về 3 giá trị
            # parsed_data_for_session chứa tất cả, bao gồm cả dữ liệu render
            # error_msg là thông báo lỗi nếu có
            # _ (raw_rows_debug) có thể bỏ qua ở đây nếu không dùng trực tiếp nữa
            parsed_data_for_session, error_msg, _ = parse_single_csv_ahp_data(file)

            if error_msg:
                flash(f"Lỗi xử lý file CSV: {error_msg}", "error")
                session.pop('csv_preview_data', None) # Xóa preview cũ nếu có lỗi
                session.pop('csv_filename', None)
                session['csv_input_error'] = True # Đặt cờ lỗi
                session.modified = True
                return redirect(url_for('upload_csv'))

            if parsed_data_for_session:
                session['csv_preview_data'] = parsed_data_for_session # Lưu toàn bộ dict mới, bao gồm cả phần render
                session['csv_filename'] = filename
                session.pop('csv_input_error', None) # Xóa cờ lỗi nếu parse thành công
                flash(f"File '{filename}' đã được đọc. Kiểm tra dữ liệu hiển thị bên dưới và xác nhận nếu chính xác.", "info")
            else: # Trường hợp hiếm: không có lỗi nhưng parsed_data là None
                flash("Không thể phân tích cú pháp file CSV. Định dạng có thể không đúng.", "error")
                session.pop('csv_preview_data', None)
                session.pop('csv_filename', None)
                session['csv_input_error'] = True
            
            session.modified = True
            return redirect(url_for('upload_csv')) # Redirect để hiển thị preview với dữ liệu mới trong session
        else:
            flash("Loại file không được phép. Chỉ chấp nhận file .csv", "error")
            session.pop('csv_preview_data', None)
            session.pop('csv_filename', None)
            session['csv_input_error'] = True
            session.modified = True
            return redirect(url_for('upload_csv'))

    # --- GET request ---
    # Lấy dữ liệu từ session để hiển thị
    preview_data_from_session = session.get('csv_preview_data')
    csv_filename = session.get('csv_filename')
    csv_input_error = session.get('csv_input_error', False) # Mặc định là False
    
    criteria_names_render = None
    alternatives_names_render = None
    criteria_preview_matrix_render = None
    alternatives_preview_matrices_render = None
    preview_data_exists = False

    # Chỉ chuẩn bị dữ liệu render nếu có preview_data và không có lỗi input nghiêm trọng
    # (csv_input_error=True có thể vẫn cho phép hiển thị preview nếu parse_single_csv_ahp_data trả về một phần dữ liệu)
    if preview_data_from_session: #and not csv_input_error: # Bỏ csv_input_error ở đây để vẫn có thể hiện thị nếu có lỗi nhẹ
        criteria_names_render = preview_data_from_session.get('criteria_names')
        alternatives_names_render = preview_data_from_session.get('alternatives_names')
        criteria_preview_matrix_render = preview_data_from_session.get('criteria_preview_matrix_render')
        alternatives_preview_matrices_render = preview_data_from_session.get('alternatives_preview_matrices_render')
        preview_data_exists = True # Đặt cờ này để biết có nên hiển thị nút Confirm hay không

    # Nếu có lỗi input và không có dữ liệu preview nào cả (ví dụ file rỗng, sai định dạng nặng)
    # thì đảm bảo không có gì được render cho phần preview
    if csv_input_error and not preview_data_exists:
        criteria_names_render = None
        alternatives_names_render = None
        criteria_preview_matrix_render = None
        alternatives_preview_matrices_render = None
        csv_filename = None # Xóa cả tên file nếu lỗi nặng không parse được gì

    return render_template("upload_csv.html",
                           preview_data_exists=preview_data_exists,
                           csv_filename=csv_filename,
                           csv_input_error=csv_input_error,
                           criteria_names_render=criteria_names_render,
                           alternatives_names_render=alternatives_names_render,
                           criteria_preview_matrix_render=criteria_preview_matrix_render,
                           alternatives_preview_matrices_render=alternatives_preview_matrices_render
                           )

@app.route("/calculate_results")
def calculate_results():
    # ... (Validation checks hiện tại giữ nguyên) ...
    if not session.get('criteria_comparison_done'):
        flash("So sánh tiêu chí chưa hoàn thành hoặc CR không hợp lệ.", "warning")
        return redirect(url_for('compare_criteria'))
    num_criteria_val = len(session.get('selected_criteria', []))
    current_alt_index_val = session.get('current_alt_criterion_index', 0)
    if not session.get('alternative_comparisons_done'):
        if current_alt_index_val >= num_criteria_val and num_criteria_val > 0 : # Allow if num_criteria is 0 (e.g. from CSV direct to results)
            session['alternative_comparisons_done'] = True
            session.modified = True
        else:
            flash(f"So sánh phương án chưa hoàn thành (đang ở tiêu chí {current_alt_index_val+1}/{num_criteria_val}).", "warning")
            return redirect(url_for('compare_alternatives'))

    num_alternatives = len(session.get('session_alternatives', []))
    num_criteria = len(session.get('selected_criteria', [])) # Lấy lại ở đây cho đúng ngữ cảnh
    required_keys = {
        'crit_weights': (list, num_criteria), 'alt_weights_all': (list, num_criteria),
        'session_alternatives': (list, num_alternatives), 'selected_criteria': (list, num_criteria),
        'crit_matrix': (list, num_criteria), 'crit_lambda_max': ((float, int), None),
        'crit_ci': ((float, int), None), 'crit_cr': ((float, int, type(None)), None),
        'crit_ri': ((float, int, type(None)), None),
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
        if data is not None:
            if isinstance(expected_type, tuple):
                if not isinstance(data, expected_type):
                    type_names = ', '.join(t.__name__ for t in expected_type)
                    missing_or_invalid.append(f"'{key}' sai kiểu (cần {type_names}, tìm thấy {type(data).__name__})")
            elif not isinstance(data, expected_type):
                missing_or_invalid.append(f"'{key}' sai kiểu (cần {expected_type.__name__}, tìm thấy {type(data).__name__})")
        if expected_len is not None and isinstance(data, list):
            if len(data) != expected_len:
                 missing_or_invalid.append(f"'{key}' sai độ dài (cần {expected_len}, tìm thấy {len(data)})")
            elif key.endswith('_all') and any(item is None for item in data):
                 missing_indices = [i for i, item in enumerate(data) if item is None]
                 missing_or_invalid.append(f"'{key}' thiếu dữ liệu tại chỉ số: {missing_indices}")
    if missing_or_invalid:
        error_message = "Dữ liệu session không đầy đủ/hợp lệ để tính kết quả: " + "; ".join(missing_or_invalid) + ". Vui lòng thử lại từ đầu."
        flash(error_message, "error")
        print("DEBUG: Session validation failed in calculate_results:", missing_or_invalid)
        # clear_session_data() # Cân nhắc, nếu lỗi do CSV thì người dùng có thể muốn sửa CSV
        return redirect(url_for('select_alternatives')) # Hoặc upload_csv nếu lỗi từ đó

    final_scores_dict = {}
    results_display = []
    best_alternative_info = None
    calculation_error = None

    try:
        # ... (Phần tính toán điểm giữ nguyên) ...
        crit_weights_np = np.array(session['crit_weights'], dtype=float) # Đổi tên biến để tránh xung đột
        alt_weights_all_list = session['alt_weights_all']
        alt_weights_matrix = np.array(alt_weights_all_list, dtype=float).T

        if crit_weights_np.shape != (num_criteria,): raise ValueError(f"Kích thước trọng số tiêu chí sai ({crit_weights_np.shape})")
        if alt_weights_matrix.shape != (num_alternatives, num_criteria): raise ValueError(f"Kích thước ma trận trọng số PA sai ({alt_weights_matrix.shape})")
        if np.isnan(crit_weights_np).any() or np.isinf(crit_weights_np).any(): raise ValueError("NaN/Inf trong trọng số tiêu chí.")
        if np.isnan(alt_weights_matrix).any() or np.isinf(alt_weights_matrix).any(): raise ValueError("NaN/Inf trong ma trận trọng số PA.")

        if abs(np.sum(crit_weights_np) - 1.0) > 1e-4: flash(f"Cảnh báo: Tổng trọng số tiêu chí ~ {np.sum(crit_weights_np):.6f}", "warning")
        col_sums = np.sum(alt_weights_matrix, axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-4):
             bad_cols = np.where(np.abs(col_sums - 1.0) > 1e-4)[0]
             flash(f"Cảnh báo: Tổng trọng số PA theo cột tiêu chí không bằng 1 (lỗi ở cột chỉ số: {bad_cols}, tổng: {col_sums[bad_cols]:.4f}).", "warning")

        final_scores_vector = np.dot(alt_weights_matrix, crit_weights_np)

        if final_scores_vector.shape != (num_alternatives,): raise ValueError(f"Kích thước vector điểm cuối sai ({final_scores_vector.shape})")
        if abs(np.sum(final_scores_vector) - 1.0) > 1e-4: flash(f"Cảnh báo: Tổng điểm cuối cùng ~ {np.sum(final_scores_vector):.6f}", "warning")

        alternatives_session = session['session_alternatives']
        final_scores_python = [float(score) for score in final_scores_vector]

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
         flash(calculation_error, "error"); print(f"Final Calc Error: {e}"); traceback.print_exc()
    except Exception as e:
         calculation_error = f"Lỗi không mong muốn trong tính toán cuối cùng: {e}"
         flash(calculation_error, "error"); print(f"Unexpected Final Calc Error: {e}"); traceback.print_exc()

    # --- Lưu vào DB (giữ nguyên logic hiện tại) ---
    can_save_to_db = session.get('all_db_alternatives', False) and session.get('all_db_criteria', False)
    save_attempted = False
    save_successful = False
    if not calculation_error and can_save_to_db and results_display:
        save_attempted = True
        conn = get_connection()
        if conn:
            analysis_group_id = str(uuid.uuid4())
            timestamp = datetime.now()
            length_mismatch_db = False
            try:
                with conn.cursor() as cursor:
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
                         final_score = result['score']
                         is_best = result['is_best']
                         if alt_id is not None:
                             alt_values_to_insert.append((analysis_group_id, timestamp, alt_id, alt_name, final_score, is_best))
                    insert_crit_query = """
                        INSERT INTO ket_qua (analysis_group_id, thoi_gian, phuong_an_id, phuong_an_ten,
                                             tieu_chi_id, tieu_chi_ten, is_alternative, is_db_source,
                                             final_score, is_best, criterion_weight)
                        VALUES (%s, %s, NULL, '', %s, %s, FALSE, TRUE, NULL, NULL, %s)
                    """
                    crit_values_to_insert = []
                    db_criteria = session.get('selected_criteria', [])
                    db_crit_weights_list = session.get('crit_weights', [])
                    if len(db_criteria) == len(db_crit_weights_list):
                        for i, crit in enumerate(db_criteria):
                            crit_id = crit.get('id')
                            crit_name = crit.get('ten_tieu_chi')
                            try: crit_weight = float(db_crit_weights_list[i])
                            except (ValueError, TypeError): continue
                            if crit_id is not None:
                                 crit_values_to_insert.append((analysis_group_id, timestamp, crit_id, crit_name, crit_weight))
                    else:
                        length_mismatch_db = True
                        flash("Lỗi lưu DB: Số lượng tiêu chí và trọng số không khớp.", "error")
                    db_insert_error = None
                    if alt_values_to_insert:
                       try: cursor.executemany(insert_alt_query, alt_values_to_insert)
                       except psycopg2.Error as alt_err: db_insert_error = f"Lỗi chèn điểm PA: {alt_err}"
                    if crit_values_to_insert and not length_mismatch_db and not db_insert_error:
                        try: cursor.executemany(insert_crit_query, crit_values_to_insert)
                        except psycopg2.Error as crit_err: db_insert_error = f"Lỗi chèn trọng số TC: {crit_err}"
                    if not db_insert_error and not length_mismatch_db and alt_values_to_insert and crit_values_to_insert:
                       conn.commit(); save_successful = True
                       flash("Kết quả đã được lưu vào cơ sở dữ liệu.", "success")
                    else:
                       conn.rollback(); save_successful = False
                       if db_insert_error: flash(f"Lưu DB thất bại: {db_insert_error}", "error")
                       elif length_mismatch_db: pass # Đã flash
                       elif not alt_values_to_insert: flash("Lưu DB thất bại: Không có dữ liệu PA hợp lệ.", "warning")
                       elif not crit_values_to_insert: flash("Lưu DB thất bại: Không có dữ liệu TC hợp lệ.", "warning")
                       else: flash("Lưu DB thất bại. Giao dịch rollback.", "warning")
            except (psycopg2.Error, Exception) as e:
                if conn: conn.rollback()
                flash(f"Lỗi lưu DB: {e}", "error"); print(f"DB Save Error: {e}"); traceback.print_exc()
                save_successful = False
            finally:
                if conn: conn.close()
        else: # get_connection failed
            save_attempted = True; save_successful = False
    elif not can_save_to_db and not calculation_error:
        flash("Kết quả không lưu vào DB vì sử dụng dữ liệu tùy chỉnh.", "info")
    elif not results_display and not calculation_error:
        flash("Không có kết quả cuối cùng để hiển thị/lưu.", "warning")


    # --- THÊM MỚI: Chuẩn bị dữ liệu cho biểu đồ ---
    chart_data = {}
    if not calculation_error and results_display:
        # 1. Biểu đồ trọng số tiêu chí (Bar Chart)
        crit_names_for_chart = [c.get('ten_tieu_chi', f'TC {i+1}') for i, c in enumerate(session.get('selected_criteria', []))]
        crit_weights_for_chart = session.get('crit_weights', [])
        chart_data['criteria_weights'] = {
            "labels": crit_names_for_chart,
            "data": crit_weights_for_chart
        }

        # 2. Biểu đồ điểm tổng hợp phương án (Horizontal Bar Chart)
        # results_display đã được sắp xếp, có thể dùng trực tiếp
        alt_names_for_chart = [r['name'] for r in results_display]
        alt_scores_for_chart = [r['score'] for r in results_display]
        chart_data['final_scores'] = {
            "labels": alt_names_for_chart,
            "data": alt_scores_for_chart
        }

        # 3. Biểu đồ trọng số phương án theo từng tiêu chí (Stacked Bar hoặc Grouped Bar)
        # Đây là phần phức tạp hơn. Cần ma trận (phương án x tiêu chí) của trọng số cục bộ.
        # alt_weights_all là list of lists: [[w_alt1_crit1, w_alt2_crit1,...], [w_alt1_crit2, w_alt2_crit2,...], ...]
        # Cần chuyển vị (transpose) nó.
        alt_weights_matrix_for_chart = []
        if session.get('alt_weights_all') and num_alternatives > 0 and num_criteria > 0:
            # session['alt_weights_all'] có dạng [crit_idx][alt_idx]
            # cần [alt_idx][crit_idx] cho stacked bar chart datasets
            temp_matrix = np.array(session.get('alt_weights_all', [])).T.tolist() # Transpose
            # Mỗi phần tử của temp_matrix bây giờ là một list trọng số của một phương án qua các tiêu chí
            # [{label: 'PA1', data: [w_pa1tc1, w_pa1tc2,...]}, {label: 'PA2', data: [w_pa2tc1, ... ]}]
            # Hoặc cho grouped bar: labels là tên tiêu chí, datasets là các phương án
            # labels: [crit1, crit2, ...]
            # datasets: [{label: PA1, data:[w_pa1c1, w_pa1c2,...]}, {label:PA2, data:[w_pa2c1, w_pa2c2,...]}]
            
            # Dùng cho grouped bar (dễ nhìn hơn khi nhiều phương án)
            datasets_alt_by_crit = []
            all_alt_names = [a.get('ten_phuong_an', f'PA {i+1}') for i, a in enumerate(session.get('session_alternatives', []))]

            if len(all_alt_names) == len(temp_matrix): # Đảm bảo kích thước khớp
                 for alt_idx, alt_name in enumerate(all_alt_names):
                    datasets_alt_by_crit.append({
                        "label": alt_name,
                        "data": temp_matrix[alt_idx] # temp_matrix[alt_idx] là list trọng số của PA này qua các tiêu chí
                    })

            chart_data['alternative_weights_by_criteria'] = {
                "labels": crit_names_for_chart, # Tiêu chí trên trục X
                "datasets": datasets_alt_by_crit
            }


    intermediate_results = get_intermediate_results_for_display()
    return render_template("results.html",
                           results=results_display,
                           intermediate=intermediate_results,
                           best_alternative_info=best_alternative_info,
                           save_attempted=save_attempted,
                           save_successful=save_successful,
                           can_save_to_db=can_save_to_db,
                           error=calculation_error,
                           chart_data=chart_data # <--- THÊM MỚI
                           )

def get_intermediate_results_for_display():
    # ... (Giữ nguyên)
    intermediate = {}
    try:
        crit_keys = ['selected_criteria', 'crit_matrix', 'crit_weights', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri']
        alt_keys = ['session_alternatives', 'alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all']
        for key in crit_keys + alt_keys:
            intermediate[key] = session.get(key)
        num_crit_check = len(intermediate.get('selected_criteria', [])) if isinstance(intermediate.get('selected_criteria'), list) else 0
        alt_lists_to_check = ['alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all']
        for key in alt_lists_to_check:
            data = intermediate.get(key)
            if not isinstance(data, list):
                 flash(f"Cảnh báo hiển thị: Dữ liệu '{key}' không phải list.", "warning")
                 intermediate[key] = [None] * num_crit_check
            elif len(data) != num_crit_check:
                 flash(f"Cảnh báo hiển thị: Dữ liệu '{key}' độ dài ({len(data)}) không khớp ({num_crit_check}).", "warning")
                 intermediate[key] = (data + [None] * num_crit_check)[:num_crit_check]
            # Bỏ qua check any(item is None) vì có thể None do lỗi CR/RI
    except Exception as e:
         flash(f"Lỗi chuẩn bị dữ liệu trung gian: {e}", "warning"); print(f"Error prep intermediate: {e}"); traceback.print_exc()
         intermediate = {}
    return intermediate

@app.route("/results_history")
def results_history():
    # ... (Giữ nguyên)
    grouped_history = {}
    db_error = None
    group_query = """
        SELECT DISTINCT analysis_group_id, MAX(thoi_gian) as analysis_time
        FROM ket_qua WHERE is_db_source = TRUE GROUP BY analysis_group_id
        ORDER BY analysis_time DESC LIMIT 20 """
    analysis_groups = execute_query(group_query, fetchall=True)
    if analysis_groups is None: db_error = "Lỗi lấy nhóm phân tích."; analysis_groups = []
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
                        'group_id': group_id, 'alternatives': [], 'criteria': [] }
                    alt_hist_query = "SELECT phuong_an_ten, final_score, is_best FROM ket_qua WHERE analysis_group_id = %s AND is_alternative = TRUE AND is_db_source = TRUE ORDER BY final_score DESC"
                    cursor.execute(alt_hist_query, (group_id,))
                    group_data['alternatives'] = [dict(row) for row in cursor.fetchall()]
                    crit_hist_query = "SELECT tieu_chi_ten, criterion_weight FROM ket_qua WHERE analysis_group_id = %s AND is_alternative = FALSE AND is_db_source = TRUE ORDER BY criterion_weight DESC"
                    cursor.execute(crit_hist_query, (group_id,))
                    group_data['criteria'] = [dict(row) for row in cursor.fetchall()]
                    if group_data['alternatives'] or group_data['criteria']:
                         grouped_history[group_id] = group_data
        except (psycopg2.Error, Exception) as e:
             db_error = f"Lỗi lấy chi tiết lịch sử: {e}"; flash(db_error, "error"); print(f"DB Hist Detail Err: {e}"); traceback.print_exc()
        finally:
            if conn_hist: conn_hist.close()
    elif not conn_hist and analysis_groups:
         db_error = "Không thể kết nối DB lấy chi tiết."; flash(db_error, "error")
    sorted_history_list = sorted(grouped_history.values(), key=lambda item: item['timestamp_obj'], reverse=True)
    return render_template("results_history.html", history_list=sorted_history_list, db_error=db_error)


# --- Error Handlers (Giữ nguyên) ---
@app.errorhandler(404)
def page_not_found(e):
     flash("Trang yêu cầu không được tìm thấy (404).", "error")
     return render_template('error.html', message='Trang không tìm thấy (404)'), 404
@app.errorhandler(500)
def internal_server_error(e):
     print(f"Internal Server Error: {e}"); traceback.print_exc()
     flash("Lỗi máy chủ nội bộ (500). Vui lòng thử lại hoặc bắt đầu lại.", "error")
     return render_template('error.html', message='Lỗi Máy chủ Nội bộ (500)'), 500
@app.errorhandler(413)
def request_entity_too_large(e):
    max_size_mb = app.config.get('MAX_CONTENT_LENGTH', 1*1024*1024) / (1024*1024)
    flash(f"File tải lên quá lớn. Giới hạn {max_size_mb:.1f}MB.", "error")
    referer = request.headers.get("Referer")
    if referer:
        if url_for('upload_csv') in referer: return redirect(url_for('upload_csv'))
        if url_for('compare_alternatives') in referer: return redirect(url_for('compare_alternatives'))
        if url_for('compare_criteria') in referer: return redirect(url_for('compare_criteria'))
    if 'current_alt_criterion_index' in session: return redirect(url_for('compare_alternatives'))
    if 'criteria_selected' in session: return redirect(url_for('compare_criteria'))
    return redirect(url_for('select_alternatives'))

# --- Main Execution (Giữ nguyên) ---
if __name__ == "__main__":
    # ... (Phần kiểm tra DB giữ nguyên) ...
    print("Kiểm tra kết nối cơ sở dữ liệu PostgreSQL...")
    conn_test = get_connection()
    if conn_test is None:
         print("\n*** CẢNH BÁO: Không thể kết nối đến cơ sở dữ liệu PostgreSQL! ***")
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

