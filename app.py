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
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

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
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("Database connection error: DATABASE_URL environment variable not set.")
        flash("Lỗi cấu hình: Không tìm thấy chuỗi kết nối cơ sở dữ liệu.", "error")
        return None
    try:
        conn = psycopg2.connect(database_url) # Use DATABASE_URL
        # Optional: Set client encoding if needed (usually utf8 is default)
        # conn.set_client_encoding('UTF8')
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        # Don't flash here as it might be called frequently. Log instead.
        # flash(f"Lỗi kết nối Database: {e}", "error") # Avoid flashing in get_connection
        return None
    except Exception as e:
        print(f"Unexpected error getting connection: {e}")
        return None

def execute_query(query, params=None, fetchone=False, fetchall=False, commit=False):
    """ Executes a query with error handling and connection management. """
    conn = get_connection()
    if not conn:
        flash("Không thể kết nối tới cơ sở dữ liệu.", "error")
        return None # Or raise an exception

    result = None
    try:
        # Use DictCursor to get results as dictionaries
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(query, params)
            if fetchone:
                result = cursor.fetchone()
            elif fetchall:
                result = cursor.fetchall()

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

# --- compute_pairwise_matrix, parse_excel_matrix, ahp_weighting remain unchanged ---
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
    # Allow common Excel extensions
    allowed_extensions = ('.xlsx', '.xls', '.xlsm', '.xlsb') # Add more if needed
    if not file_storage.filename.lower().endswith(allowed_extensions):
        return None, "Định dạng file không hợp lệ. Chỉ chấp nhận file Excel (ví dụ: .xlsx, .xls)."

    try:
        # Read only the first sheet using openpyxl engine for broader compatibility
        df = pd.read_excel(file_storage, header=None, engine='openpyxl')

        # Attempt to find the start of the numeric matrix
        start_row, start_col = -1, -1
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                cell_value = df.iloc[r, c]
                # Check if it's numeric (int, float, or numpy numeric types)
                is_numeric = isinstance(cell_value, (int, float, np.number))
                if is_numeric:
                     # Check if this looks like the top-left '1'
                    if abs(float(cell_value) - 1.0) < 1e-6:
                        # Look ahead to see if it's likely a matrix
                        if r + 1 < df.shape[0] and c + 1 < df.shape[1]:
                             next_cell = df.iloc[r+1, c+1]
                             if isinstance(next_cell, (int, float, np.number)):
                                start_row, start_col = r, c
                                break
            if start_row != -1:
                break

        if start_row == -1 or start_col == -1:
            return None, "Không thể tự động xác định ma trận số trong file Excel. Đảm bảo ma trận bắt đầu bằng số 1 ở góc trên bên trái và chứa các giá trị số."

        # Extract the potential matrix based on expected size
        if start_row + expected_size > df.shape[0] or start_col + expected_size > df.shape[1]:
            return None, f"Kích thước ma trận số tìm thấy không đủ lớn. Cần ma trận {expected_size}x{expected_size} bắt đầu từ ô ({start_row+1},{start_col+1})."

        matrix_df = df.iloc[start_row : start_row + expected_size, start_col : start_col + expected_size]

        # Convert to numeric, coercing errors
        matrix_np = matrix_df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

        if np.isnan(matrix_np).any():
            # Find first NaN location for better error message
            nan_loc = np.argwhere(np.isnan(matrix_np))
            first_nan_row, first_nan_col = nan_loc[0]
            return None, f"Ma trận chứa giá trị không phải số tại vị trí ({start_row + first_nan_row + 1},{start_col + first_nan_col + 1}) trong file Excel. Vui lòng kiểm tra file."


        # --- Basic Matrix Validation ---
        if matrix_np.shape != (expected_size, expected_size):
            return None, f"Kích thước ma trận không đúng. Cần {expected_size}x{expected_size}, tìm thấy {matrix_np.shape}."

        if not np.allclose(np.diag(matrix_np), 1.0):
            return None, "Đường chéo chính của ma trận phải bằng 1."

        for i in range(expected_size):
            for j in range(i + 1, expected_size):
                 val_ij = matrix_np[i, j]
                 val_ji = matrix_np[j, i]
                 if val_ij <= 0 or val_ji <= 0:
                     return None, f"Giá trị tại ({start_row+i+1},{start_col+j+1}) hoặc ({start_row+j+1},{start_col+i+1}) không phải là số dương."
                 if abs(val_ij * val_ji - 1.0) > 1e-6: # Check reciprocal relationship
                    return None, f"Giá trị nghịch đảo không chính xác tại vị trí ({start_row+i+1},{start_col+j+1}) và ({start_row+j+1},{start_col+i+1}). Giá trị phải dương và A[j,i] = 1/A[i,j] (tìm thấy {val_ij:.3f} và {val_ji:.3f})."

        return matrix_np, None # Success

    except ValueError as e: # Catch specific pandas errors if possible
        traceback.print_exc()
        return None, f"Lỗi giá trị khi đọc file Excel: {e}"
    except ImportError:
         return None, "Lỗi thiếu thư viện đọc file Excel. Cần cài đặt 'openpyxl'."
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

    try:
        eigvals, eigvecs = np.linalg.eig(matrix)
        real_eigvals = np.real(eigvals)
        lambda_max = np.max(real_eigvals)

        max_eigval_idx = np.argmax(real_eigvals)
        principal_eigvec = np.real(eigvecs[:, max_eigval_idx])

        if np.all(principal_eigvec <= 1e-9): # Handle all-negative/zero case
            # This is unusual, might indicate a problem matrix
             principal_eigvec = np.abs(principal_eigvec) # Try absolute values
             if np.all(principal_eigvec <= 1e-9): # Still zero? Fallback
                 flash("Cảnh báo: Vector trọng số chính gần như bằng không. Sử dụng trọng số bằng nhau làm dự phòng.", "warning")
                 weights = np.ones(n) / n
             else:
                weights = principal_eigvec
        elif np.any(principal_eigvec < -1e-9):
             flash("Cảnh báo: Vector trọng số có giá trị âm, có thể do tính không nhất quán cao.", "warning")
             weights = np.maximum(0, principal_eigvec) # Set negative values to 0
        else:
             weights = principal_eigvec

        sum_weights = np.sum(weights)

        if abs(sum_weights) < 1e-9:
             flash("Lỗi: Tổng vector trọng số gần bằng 0. Không thể chuẩn hóa. Sử dụng trọng số bằng nhau làm dự phòng.", "error")
             weights = np.ones(n) / n # Fallback to equal weights
             # Recalculate consistency based on original matrix, even if weights are fallback
             lambda_max_for_ci = lambda_max # Use the calculated one
        else:
             weights /= sum_weights # Normalize
             lambda_max_for_ci = lambda_max # Use the calculated one

        # Calculate Consistency
        if n > 2:
            CI = (lambda_max_for_ci - n) / (n - 1)
            CI = max(0.0, CI)
            RI = RI_DICT.get(n)
            if RI is None:
                 flash(f"Lỗi: Không tìm thấy Chỉ số ngẫu nhiên (RI) cho ma trận kích thước n={n}.", "error")
                 RI = RI_DICT[max(RI_DICT.keys())] # Fallback might be misleading
                 CR = float('inf')
            elif RI == 0:
                 CR = float('inf') if CI > 1e-9 else 0.0
            else:
                 CR = CI / RI
        else: # n <= 2
            CI = 0.0
            RI = 0.00
            CR = 0.0

        if any(x is not None and (math.isnan(x) or math.isinf(x)) for x in [lambda_max_for_ci, CI, CR]) or \
           any(np.isnan(weights)) or any(np.isinf(weights)):
            flash("Lỗi: Kết quả tính toán AHP chứa NaN hoặc vô cực.", "error")
            print(f"NaN/Inf detected: lambda_max={lambda_max_for_ci}, CI={CI}, CR={CR}, weights={weights}")
            # Fallback or signal error
            return None, None, None, None, None # Signal error

        # Round results for cleaner storage/display
        weights = np.round(weights, 6)
        lambda_max_for_ci = round(lambda_max_for_ci, 6)
        CI = round(CI, 6)
        CR = round(CR, 6)
        # RI doesn't need rounding usually

        # Ensure weights still sum reasonably close to 1 after rounding
        if abs(np.sum(weights) - 1.0) > 1e-5:
            # Re-normalize slightly if rounding caused drift
            weights /= np.sum(weights)
            weights = np.round(weights, 6)


        return weights, lambda_max_for_ci, CI, CR, RI

    except np.linalg.LinAlgError as e:
        flash(f"Lỗi tính toán đại số tuyến tính: {e}. Ma trận có thể không hợp lệ.", "error")
        traceback.print_exc()
        return None, None, None, None, None
    except Exception as e:
        flash(f"Lỗi không mong muốn trong tính toán AHP: {e}", "error")
        traceback.print_exc()
        return None, None, None, None, None

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
        all_db_alternatives = False # Default

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

            # PostgreSQL uses %s placeholders
            format_strings = ','.join(['%s'] * len(selected_ids))
            # No direct FIELD equivalent, fetch and reorder in Python
            query = f"SELECT id, ten_phuong_an FROM phuong_an WHERE id IN ({format_strings})"
            # Parameters need to be a tuple
            params = tuple(selected_ids)
            alternatives_from_db_unordered = execute_query(query, params, fetchall=True)

            if alternatives_from_db_unordered is None: # Check if query failed
                 # execute_query already flashed the error
                 return redirect(url_for('select_alternatives'))

            if len(alternatives_from_db_unordered) != len(selected_ids):
                 flash("Không thể truy xuất tất cả phương án đã chọn hoặc ID không tồn tại.", "error")
                 return redirect(url_for('select_alternatives'))

            # Reorder based on original selection order
            db_map = {item['id']: item for item in alternatives_from_db_unordered}
            alternatives = [db_map[sid] for sid in selected_ids if sid in db_map]

            if len(alternatives) != len(selected_ids):
                 # This case should ideally not happen if the first length check passed
                 flash("Lỗi sắp xếp lại phương án đã chọn.", "error")
                 return redirect(url_for('select_alternatives'))

            all_db_alternatives = True # Success!


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

        clear_session_data() # Clear previous run data
        session['session_alternatives'] = alternatives
        session['all_db_alternatives'] = all_db_alternatives
        session['alternatives_selected'] = True
        session.modified = True

        return redirect(url_for('select_criteria'))

    # --- GET Request ---
    clear_session_data() # Start fresh on GET
    db_error = None
    query = "SELECT id, ten_phuong_an FROM phuong_an ORDER BY id"
    all_alternatives_db = execute_query(query, fetchall=True)

    # execute_query handles flashing errors, but we might set a db_error flag if needed
    if all_alternatives_db is None:
         db_error = "Lỗi lấy danh sách phương án từ DB."
         # flash message is already handled by execute_query

    return render_template("select_alternatives.html",
                           all_alternatives_db=all_alternatives_db if all_alternatives_db else [], # Ensure it's a list
                           db_error=db_error)

@app.route("/select_criteria", methods=["GET", "POST"])
def select_criteria():
    """Step 1: User selects or enters criteria (min MIN_CRITERIA)."""
    if not session.get('alternatives_selected'):
        flash("Vui lòng chọn hoặc nhập các phương án trước.", "info")
        return redirect(url_for('select_alternatives'))

    if request.method == "POST":
        selection_mode = request.form.get('mode')
        criteria = []
        all_db_criteria = False # Default

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
            # Fetch and reorder in Python
            query = f"SELECT id, ten_tieu_chi FROM tieu_chi WHERE id IN ({format_strings})"
            params = tuple(selected_ids)
            criteria_from_db_unordered = execute_query(query, params, fetchall=True)

            if criteria_from_db_unordered is None:
                 return redirect(url_for('select_criteria')) # Error flashed

            if len(criteria_from_db_unordered) != len(selected_ids):
                 flash("Không thể truy xuất tất cả tiêu chí đã chọn hoặc ID không tồn tại.", "error")
                 return redirect(url_for('select_criteria'))

            # Reorder
            db_map = {item['id']: item for item in criteria_from_db_unordered}
            criteria = [db_map[sid] for sid in selected_ids if sid in db_map]

            if len(criteria) != len(selected_ids):
                 flash("Lỗi sắp xếp lại tiêu chí đã chọn.", "error")
                 return redirect(url_for('select_criteria'))

            all_db_criteria = True # Success!


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

        # Clear data from subsequent steps
        clear_ahp_session_data()

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
        # Flash handled by execute_query

    return render_template("select_criteria.html",
                           all_criteria_db=all_criteria_db if all_criteria_db else [],
                           db_error=db_error,
                           selected_alternatives=session.get('session_alternatives', []))

# --- compare_criteria, compare_alternatives remain largely the same, ---
# --- as they mostly handle form data, Excel parsing, and AHP logic, not direct DB interaction here ---
@app.route("/compare_criteria", methods=["GET", "POST"])
def compare_criteria():
    """Step 2: User compares selected criteria (manual or Excel)."""
    if not session.get('criteria_selected'):
        flash("Vui lòng chọn tiêu chí trước.", "info")
        return redirect(url_for('select_criteria'))

    selected_criteria = session.get('selected_criteria', [])
    if not selected_criteria or len(selected_criteria) < MIN_CRITERIA:
        flash(f"Số lượng tiêu chí không hợp lệ ({len(selected_criteria)}), cần ít nhất {MIN_CRITERIA}. Vui lòng chọn lại.", "error")
        return redirect(url_for('select_criteria'))

    criteria_names = [c['ten_tieu_chi'] for c in selected_criteria]
    num_criteria = len(selected_criteria)
    crit_matrix = None
    input_method = "form" # Default

    if request.method == "POST":
        # Check for Excel file upload first
        if 'criteria_excel_file' in request.files:
            file = request.files['criteria_excel_file']
            if file and file.filename != '':
                # filename = secure_filename(file.filename) # Not needed if not saving
                crit_matrix, error_msg = parse_excel_matrix(file, num_criteria, criteria_names)
                if error_msg:
                    flash(f"Lỗi xử lý file Excel tiêu chí: {error_msg}", "error")
                    session.pop('form_data_crit', None)
                    session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
                    session.modified = True
                    return redirect(url_for('compare_criteria'))
                elif crit_matrix is not None:
                     input_method = "excel"
                     flash("Đã nhập ma trận so sánh tiêu chí từ file Excel.", "info")

        # If no valid matrix from Excel, try form data
        if crit_matrix is None:
            input_method = "form"
            crit_matrix = compute_pairwise_matrix("pc", criteria_names, request.form)
            if crit_matrix is None:
                session['form_data_crit'] = request.form
                session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
                session.modified = True
                return redirect(url_for('compare_criteria'))

        # --- Matrix obtained, proceed with AHP calculation ---
        crit_weights, crit_lambda_max, crit_ci, crit_cr, crit_ri = ahp_weighting(crit_matrix)

        if crit_weights is None:
            if input_method == "form":
                 session['form_data_crit'] = request.form
            session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
            session.modified = True
            return redirect(url_for('compare_criteria'))

        # Store results temporarily and permanently if CR passes
        session['crit_matrix'] = crit_matrix.tolist()
        session['crit_lambda_max'] = crit_lambda_max
        session['crit_ci'] = crit_ci
        session['crit_cr'] = crit_cr
        session['crit_ri'] = crit_ri

        if crit_cr > CR_THRESHOLD:
            flash(f"Tỷ số nhất quán (CR = {crit_cr:.4f}) vượt ngưỡng ({CR_THRESHOLD:.2f}). Vui lòng xem lại các so sánh tiêu chí.", "error")
            session['criteria_comparison_done'] = False
            if input_method == "form":
                 session['form_data_crit'] = request.form
            else:
                 session.pop('form_data_crit', None)
            session.modified = True
            return redirect(url_for('compare_criteria'))
        else:
            # --- CR is acceptable ---
            flash(f"So sánh tiêu chí thành công (CR = {crit_cr:.4f}). Tiếp tục so sánh phương án.", "success")
            session['crit_weights'] = crit_weights.tolist()
            session['criteria_comparison_done'] = True
            session.pop('form_data_crit', None)

            # Initialize structures for alternative comparisons
            num_alternatives = len(session.get('session_alternatives', []))
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
    if 'session_alternatives' not in session or 'selected_criteria' not in session:
        flash("Dữ liệu session bị thiếu (phương án/tiêu chí). Vui lòng bắt đầu lại.", "error")
        return redirect(url_for('select_alternatives'))

    selected_criteria = session['selected_criteria']
    alternatives = session['session_alternatives']

    if not alternatives or len(alternatives) < MIN_ALTERNATIVES:
         flash(f"Số lượng phương án không hợp lệ ({len(alternatives)}), cần ít nhất {MIN_ALTERNATIVES}.", "error")
         return redirect(url_for('select_alternatives'))
    if not selected_criteria:
         flash("Không có tiêu chí nào được chọn.", "error")
         return redirect(url_for('select_criteria'))

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
    alt_matrix = None
    input_method = "form"

    if request.method == "POST":
         # Trust session index primarily
         # Check for Excel file upload
         if f'alternative_excel_file_{current_index}' in request.files:
             file = request.files[f'alternative_excel_file_{current_index}']
             if file and file.filename != '':
                # filename = secure_filename(file.filename) # Not needed
                alt_matrix, error_msg = parse_excel_matrix(file, num_alternatives, alternative_names)
                if error_msg:
                    flash(f"Lỗi xử lý file Excel cho tiêu chí '{current_criterion['ten_tieu_chi']}': {error_msg}", "error")
                    session.pop('form_data_alt', None)
                    clear_temporary_alt_data_for_index(current_index) # Clear temp results
                    session.modified = True
                    return redirect(url_for('compare_alternatives'))
                elif alt_matrix is not None:
                     input_method = "excel"
                     flash(f"Đã nhập ma trận so sánh phương án cho '{current_criterion['ten_tieu_chi']}' từ file Excel.", "info")

         # If no valid matrix from Excel, try form data
         if alt_matrix is None:
            input_method = "form"
            prefix = f"alt_pc_{current_index}"
            alt_matrix = compute_pairwise_matrix(prefix, alternative_names, request.form)

            if alt_matrix is None:
                session['form_data_alt'] = request.form
                clear_temporary_alt_data_for_index(current_index) # Clear temp results
                session.modified = True
                return redirect(url_for('compare_alternatives'))

         # --- Matrix obtained, perform AHP calculation ---
         alt_weights, alt_lambda_max, alt_ci, alt_cr, alt_ri = ahp_weighting(alt_matrix)

         if alt_weights is None:
             if input_method == "form":
                  session['form_data_alt'] = request.form
             clear_temporary_alt_data_for_index(current_index) # Clear temp results
             session.modified = True
             return redirect(url_for('compare_alternatives'))

         # Store results temporarily for display if CR fails
         session[f'temp_alt_matrix_{current_index}'] = alt_matrix.tolist()
         session[f'temp_alt_lambda_max_{current_index}'] = alt_lambda_max
         session[f'temp_alt_ci_{current_index}'] = alt_ci
         session[f'temp_alt_cr_{current_index}'] = alt_cr
         session[f'temp_alt_ri_{current_index}'] = alt_ri

         if alt_cr > CR_THRESHOLD:
             flash(f"CR cho phương án theo '{current_criterion['ten_tieu_chi']}' ({alt_cr:.4f}) > {CR_THRESHOLD:.2f}. Vui lòng xem lại.", "error")
             session['alternative_comparisons_done'] = False
             if input_method == "form":
                  session['form_data_alt'] = request.form
             else:
                  session.pop('form_data_alt', None)
             session.modified = True
             # Keep temp results, redirect back to the same criterion page
             return redirect(url_for('compare_alternatives'))
         else:
             # --- Consistent! Store results permanently ---
             flash(f"So sánh phương án theo '{current_criterion['ten_tieu_chi']}' đã lưu (CR = {alt_cr:.4f}).", "success")

             def ensure_session_list(key, length, default_val=None):
                 # Ensure list exists and has correct length before assigning
                 data = session.get(key)
                 if not isinstance(data, list) or len(data) != length:
                     session[key] = [default_val] * length
                 return session[key]

             alt_matrices_all = ensure_session_list('alt_matrices_all', num_criteria, default_val=[])
             alt_weights_all = ensure_session_list('alt_weights_all', num_criteria)
             alt_lambda_max_all = ensure_session_list('alt_lambda_max_all', num_criteria)
             alt_ci_all = ensure_session_list('alt_ci_all', num_criteria)
             alt_cr_all = ensure_session_list('alt_cr_all', num_criteria)
             alt_ri_all = ensure_session_list('alt_ri_all', num_criteria)

             # Store permanent results using validated lists
             session['alt_matrices_all'][current_index] = alt_matrix.tolist()
             session['alt_weights_all'][current_index] = alt_weights.tolist()
             session['alt_lambda_max_all'][current_index] = alt_lambda_max
             session['alt_ci_all'][current_index] = alt_ci
             session['alt_cr_all'][current_index] = alt_cr
             session['alt_ri_all'][current_index] = alt_ri

             # Clear temporary data for this index and general form data
             clear_temporary_alt_data_for_index(current_index)
             session.pop('form_data_alt', None)

             # Move to the next criterion
             next_index = current_index + 1
             session['current_alt_criterion_index'] = next_index
             session.modified = True

             if next_index >= num_criteria:
                 session['alternative_comparisons_done'] = True
                 return redirect(url_for('calculate_results'))
             else:
                 return redirect(url_for('compare_alternatives')) # Redirect to GET for next

    # --- GET request ---
    form_data = session.get('form_data_alt', None)
    alt_lambda_max = session.get(f'temp_alt_lambda_max_{current_index}')
    alt_ci = session.get(f'temp_alt_ci_{current_index}')
    alt_cr = session.get(f'temp_alt_cr_{current_index}')
    alt_ri = session.get(f'temp_alt_ri_{current_index}')

    # Clear general form data after retrieving it for display
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
    """Step 4: Calculate final scores, display results, and save to DB if applicable."""
    # --- Validation Checks ---
    if not session.get('criteria_comparison_done'):
        flash("So sánh tiêu chí chưa hoàn thành hoặc CR không hợp lệ.", "warning")
        return redirect(url_for('compare_criteria'))

    num_criteria = len(session.get('selected_criteria', []))
    num_alternatives = len(session.get('session_alternatives', []))
    current_alt_index = session.get('current_alt_criterion_index', -1)

    if not session.get('alternative_comparisons_done'):
        if current_alt_index == num_criteria and num_criteria > 0:
             session['alternative_comparisons_done'] = True
             session.modified = True
        else:
             flash(f"So sánh phương án chưa hoàn thành (đang ở tiêu chí {current_alt_index+1}/{num_criteria}).", "warning")
             return redirect(url_for('compare_alternatives'))

    required_keys = [
        'crit_weights', 'alt_weights_all', 'session_alternatives', 'selected_criteria',
        'crit_matrix', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri',
        'alt_matrices_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all',
        'all_db_alternatives', 'all_db_criteria'
    ]
    missing_or_invalid = []
    for key in required_keys:
        data = session.get(key)
        if data is None:
            missing_or_invalid.append(f"'{key}' is missing")
            continue
        if key.startswith('alt_') and key.endswith('_all'):
             if not isinstance(data, list) or len(data) != num_criteria:
                 missing_or_invalid.append(f"'{key}' has incorrect length (found {len(data) if isinstance(data, list) else 'N/A'}, expected {num_criteria})")
             elif any(x is None for x in data):
                  missing_indices = [i for i, x in enumerate(data) if x is None]
                  missing_or_invalid.append(f"'{key}' has missing data at indices: {missing_indices}")

    if missing_or_invalid:
        error_message = "Dữ liệu session không đầy đủ/hợp lệ để tính kết quả: " + "; ".join(missing_or_invalid) + ". Vui lòng thử lại."
        flash(error_message, "error")
        if any('alt_' in s for s in missing_or_invalid): return redirect(url_for('compare_alternatives'))
        elif any('crit_' in s for s in missing_or_invalid): return redirect(url_for('compare_criteria'))
        else: return redirect(url_for('clear_session_and_start'))

    # --- Perform Final Calculation ---
    try:
        crit_weights = np.array(session['crit_weights'])
        alt_weights_all_list = session['alt_weights_all']
        alt_weights_matrix = np.array(alt_weights_all_list).T

        if crit_weights.shape != (num_criteria,):
            raise ValueError(f"Kích thước trọng số tiêu chí không đúng ({crit_weights.shape}), cần ({num_criteria},)")
        if alt_weights_matrix.shape != (num_alternatives, num_criteria):
             raise ValueError(f"Kích thước ma trận trọng số PA sau chuyển vị không đúng ({alt_weights_matrix.shape}), cần ({num_alternatives}, {num_criteria})")
        if np.isnan(crit_weights).any() or np.isinf(crit_weights).any(): raise ValueError("NaN/Inf trong trọng số tiêu chí.")
        if np.isnan(alt_weights_matrix).any() or np.isinf(alt_weights_matrix).any(): raise ValueError("NaN/Inf trong ma trận trọng số phương án.")
        if abs(np.sum(crit_weights) - 1.0) > 1e-5: flash(f"Cảnh báo: Tổng trọng số tiêu chí ~ {np.sum(crit_weights):.6f} (nên bằng 1).", "warning")

        # --- Final Score Calculation ---
        final_scores_vector = np.dot(alt_weights_matrix, crit_weights)

        if final_scores_vector.shape != (num_alternatives,):
            raise ValueError(f"Kích thước vector điểm cuối cùng không đúng ({final_scores_vector.shape}), cần ({num_alternatives},)")
        if abs(np.sum(final_scores_vector) - 1.0) > 1e-5:
             flash(f"Cảnh báo: Tổng điểm cuối cùng ~ {np.sum(final_scores_vector):.6f} (nên bằng 1).", "warning")

        # --- Prepare results for display ---
        alternatives_session = session['session_alternatives']
        final_scores_dict = {
            alt['ten_phuong_an']: score for alt, score in zip(alternatives_session, final_scores_vector)
        }
        best_alternative_name = max(final_scores_dict, key=final_scores_dict.get) if final_scores_dict else None

        results_display = []
        best_alternative_info = None
        if alternatives_session and final_scores_dict:
            for i, alt in enumerate(alternatives_session):
                alt_name = alt['ten_phuong_an']
                score = final_scores_vector[i]
                is_best = (alt_name == best_alternative_name)
                display_item = {
                    'id': alt.get('id'),
                    'name': alt_name,
                    'score': score,
                    'is_best': is_best
                }
                results_display.append(display_item)
                if is_best:
                    best_alternative_info = display_item

            results_display.sort(key=lambda x: x['score'], reverse=True)

        session['final_scores'] = final_scores_dict
        session['best_alternative_info'] = best_alternative_info
        session.modified = True

    except (ValueError, TypeError, IndexError) as e:
         flash(f"Lỗi trong quá trình tính toán cuối cùng: {e}", "error")
         print(f"Final Calculation Error Details: {e}"); traceback.print_exc()
         intermediate_results = get_intermediate_results_for_display()
         return render_template("results.html", error=f"Lỗi tính toán: {e}", intermediate=intermediate_results)
    except Exception as e:
         flash(f"Đã xảy ra lỗi không mong muốn trong quá trình tính toán cuối cùng: {e}.", "error")
         print(f"Unexpected Final Calculation Error: {e}"); traceback.print_exc()
         return render_template("error.html", message=f"Lỗi không mong muốn: {e}")


    # --- Save results to database ---
    can_save_to_db = session.get('all_db_alternatives', False) and session.get('all_db_criteria', False)
    save_attempted = False
    save_successful = False

    if can_save_to_db and results_display:
        save_attempted = True
        conn = get_connection() # Get a new connection for the transaction
        if conn:
            analysis_group_id = str(uuid.uuid4())
            timestamp = datetime.now()
            try:
                with conn.cursor() as cursor: # Default cursor is fine for inserts
                    # Start transaction implicitly with first execute, or explicitly:
                    # conn.autocommit = False # Ensure we control commit/rollback

                    # 1. Insert alternative scores
                    insert_alt_query = """
                        INSERT INTO ket_qua (analysis_group_id, thoi_gian, phuong_an_id, phuong_an_ten,
                                             tieu_chi_id, tieu_chi_ten, is_alternative, is_db_source,
                                             final_score, is_best, criterion_weight)
                        VALUES (%s, %s, %s, %s, NULL, '', TRUE, TRUE, %s, %s, NULL)
                    """
                    alt_values_to_insert = []
                    for result in results_display:
                         if result['id'] is not None: # Only save DB alternatives
                             alt_values_to_insert.append((
                                 analysis_group_id, timestamp, result['id'], result['name'],
                                 result['score'], result['is_best']
                             ))
                    if alt_values_to_insert:
                        # Use execute_values for potential performance gain if psycopg2 > 2.7
                        # psycopg2.extras.execute_values(cursor, insert_alt_query, alt_values_to_insert)
                        # Or stick to executemany for broader compatibility:
                         cursor.executemany(insert_alt_query, alt_values_to_insert)

                    # 2. Insert criteria weights
                    insert_crit_query = """
                        INSERT INTO ket_qua (analysis_group_id, thoi_gian, phuong_an_id, phuong_an_ten,
                                             tieu_chi_id, tieu_chi_ten, is_alternative, is_db_source,
                                             final_score, is_best, criterion_weight)
                        VALUES (%s, %s, NULL, '', %s, %s, FALSE, TRUE, NULL, NULL, %s)
                    """
                    crit_values_to_insert = []
                    db_criteria = session['selected_criteria']
                    db_crit_weights = session['crit_weights']
                    for i, crit in enumerate(db_criteria):
                        if crit.get('id') is not None: # Only save DB criteria
                             crit_values_to_insert.append((
                                 analysis_group_id, timestamp, crit['id'], crit['ten_tieu_chi'],
                                 db_crit_weights[i]
                             ))
                    if crit_values_to_insert:
                        cursor.executemany(insert_crit_query, crit_values_to_insert)

                    # Only commit if both inserts seemed okay (had data)
                    if alt_values_to_insert and crit_values_to_insert:
                        conn.commit() # <--- Commit transaction
                        save_successful = True
                        flash("Kết quả phân tích đã được lưu vào cơ sở dữ liệu.", "success")
                    else:
                         conn.rollback() # Rollback if nothing to insert
                         flash("Không có dữ liệu hợp lệ để lưu vào cơ sở dữ liệu.", "warning")


            except (psycopg2.Error, Exception) as e:
                if conn: conn.rollback() # <--- Rollback on error
                flash(f"Lỗi lưu kết quả vào cơ sở dữ liệu: {e}", "error")
                print(f"DB Save Error: {e}"); traceback.print_exc()
                save_successful = False # Ensure it's false on error
            finally:
                if conn: conn.close()
        else:
            flash("Không thể kết nối đến cơ sở dữ liệu để lưu kết quả.", "error")
            save_attempted = True
            save_successful = False

    elif not can_save_to_db:
        flash("Kết quả được tính toán nhưng không lưu vào DB vì sử dụng phương án hoặc tiêu chí tùy chỉnh.", "info")
    elif not results_display:
        flash("Không có kết quả cuối cùng để tính toán hoặc lưu.", "info")


    # --- Prepare Intermediate Results for Display ---
    intermediate_results = get_intermediate_results_for_display()

    return render_template("results.html",
                           results=results_display,
                           intermediate=intermediate_results,
                           best_alternative_info=best_alternative_info,
                           save_attempted=save_attempted,
                           save_successful=save_successful,
                           can_save_to_db=can_save_to_db)


@app.route("/results_history")
def results_history():
    """Displays recent results from the database, grouped by analysis run."""
    grouped_history = {}
    db_error = None

    # Use execute_query helper
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
        analysis_groups = [] # Ensure it's iterable

    conn_hist = get_connection() # Need a persistent connection for multiple queries
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

                    # Get alternative results for this group
                    alt_hist_query = """
                        SELECT phuong_an_ten, final_score, is_best
                        FROM ket_qua
                        WHERE analysis_group_id = %s AND is_alternative = TRUE AND is_db_source = TRUE
                        ORDER BY final_score DESC
                    """
                    cursor.execute(alt_hist_query, (group_id,))
                    group_data['alternatives'] = cursor.fetchall()

                    # Get criteria weights for this group
                    crit_hist_query = """
                        SELECT tieu_chi_ten, criterion_weight
                        FROM ket_qua
                        WHERE analysis_group_id = %s AND is_alternative = FALSE AND is_db_source = TRUE
                        ORDER BY criterion_weight DESC
                    """
                    cursor.execute(crit_hist_query, (group_id,))
                    group_data['criteria'] = cursor.fetchall()

                    if group_data['alternatives'] or group_data['criteria']:
                         grouped_history[group_id] = group_data

        except (psycopg2.Error, Exception) as e:
             db_error = f"Lỗi lấy chi tiết lịch sử: {e}"
             flash(db_error, "error"); print(f"DB History Detail Error: {e}"); traceback.print_exc()
        finally:
            if conn_hist: conn_hist.close()
    elif not conn_hist:
         db_error = "Không thể kết nối DB để lấy chi tiết lịch sử."
         flash(db_error, "error")


    sorted_history_list = sorted(grouped_history.values(), key=lambda item: item['timestamp_obj'], reverse=True)

    return render_template("results_history.html", history_list=sorted_history_list, db_error=db_error)


# --- Helper Functions --- (Largely unchanged, but added clear logic)

def get_intermediate_results_for_display():
    """Safely retrieves intermediate results from session for the results page."""
    intermediate = {}
    try:
        # Define keys expected in session
        crit_keys = ['selected_criteria', 'crit_matrix', 'crit_weights', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri']
        alt_keys = ['session_alternatives', 'alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all']

        for key in crit_keys + alt_keys:
            intermediate[key] = session.get(key) # Use get to avoid KeyError

        # Perform basic validation after retrieving all
        num_crit_check = len(intermediate.get('selected_criteria', []))
        alt_lists_to_check = ['alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all']

        valid = True
        for key in alt_lists_to_check:
            data = intermediate.get(key)
            if not isinstance(data, list) or len(data) != num_crit_check:
                 flash(f"Cảnh báo hiển thị: Dữ liệu '{key}' không hợp lệ hoặc độ dài không khớp.", "warning")
                 intermediate[key] = [None] * num_crit_check # Attempt to pad for display
                 valid = False
            elif any(x is None for x in data): # Check for None placeholders within the list
                 flash(f"Cảnh báo hiển thị: Dữ liệu '{key}' còn thiếu sót.", "warning")
                 # No need to pad here, just warn

    except Exception as e:
         flash(f"Lỗi khi chuẩn bị dữ liệu trung gian để hiển thị: {e}", "warning")
         print(f"Error preparing intermediate results: {e}")
         traceback.print_exc()
         intermediate = {} # Reset on error
    return intermediate

def clear_temporary_alt_data_for_index(index):
     """Clears temporary session keys for a specific alt comparison index."""
     keys = ['temp_alt_matrix', 'temp_alt_lambda_max', 'temp_alt_ci', 'temp_alt_cr', 'temp_alt_ri']
     for key_base in keys:
         session.pop(f'{key_base}_{index}', None)

def clear_temporary_alt_data(num_criteria):
     """Clears all temporary alt comparison keys."""
     max_crit_guess = max(num_criteria if isinstance(num_criteria, int) and num_criteria > 0 else 0, 25) # Safe upper bound
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
    num_crit_guess = len(session.get('selected_criteria', [])) # Get length before clearing
    for key in keys_to_clear:
        session.pop(key, None)
    clear_temporary_alt_data(num_crit_guess) # Clear indexed temp keys
    session.modified = True

def clear_session_data():
    """Clears ALL session data related to an AHP run, including alternatives."""
    clear_ahp_session_data() # Clear steps 1 onwards
    session.pop('session_alternatives', None)
    session.pop('all_db_alternatives', None)
    session.pop('alternatives_selected', None)
    session.modified = True

@app.route("/clear")
def clear_session_and_start():
    """Clears the session and redirects to the start."""
    clear_session_data()
    flash("Session đã được xóa. Bắt đầu một phân tích mới.", "info")
    return redirect(url_for('select_alternatives'))

# --- Error Handlers --- (Unchanged, but good to keep)
@app.errorhandler(404)
def page_not_found(e):
     flash("Trang yêu cầu không được tìm thấy (404).", "error")
     return render_template('error.html', message='Trang không tìm thấy (404)'), 404

@app.errorhandler(500)
def internal_server_error(e):
     print(f"Internal Server Error: {e}")
     traceback.print_exc()
     flash("Đã xảy ra lỗi máy chủ nội bộ (500). Vui lòng thử lại.", "error")
     return render_template('error.html', message='Lỗi Máy chủ Nội bộ (500)'), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    flash("File tải lên quá lớn. Vui lòng chọn file nhỏ hơn (giới hạn 1MB).", "error")
    # Attempt to redirect back intelligently
    referer = request.headers.get("Referer")
    if referer:
        # Basic check to see if it was likely an upload page
        if 'compare_alternatives' in referer:
            return redirect(url_for('compare_alternatives'))
        if 'compare_criteria' in referer:
            return redirect(url_for('compare_criteria'))
    # Fallback redirect
    if 'current_alt_criterion_index' in session:
        return redirect(url_for('compare_alternatives'))
    elif 'criteria_selected' in session:
        return redirect(url_for('compare_criteria'))
    else:
        return redirect(url_for('select_criteria'))


# --- Main Execution ---
if __name__ == "__main__":
    # Test DB connection on startup using the new function
    print("Kiểm tra kết nối cơ sở dữ liệu PostgreSQL...")
    conn_test = get_connection()
    if conn_test is None:
         print("\n*** CẢNH BÁO: Không thể kết nối đến cơ sở dữ liệu PostgreSQL! ***")
         print("1. Đảm bảo biến môi trường DATABASE_URL được đặt chính xác (trong .env hoặc hệ thống).")
         print("2. Kiểm tra thông tin đăng nhập/host/port trong DATABASE_URL.")
         print("3. Đảm bảo PostgreSQL server đang chạy và chấp nhận kết nối.")
         print("4. Chạy lệnh SQL (cung cấp riêng) để tạo bảng nếu chưa có.\n")
    else:
        print("Kết nối cơ sở dữ liệu PostgreSQL thành công.")
        conn_test.close()

    # Get port from environment variable for Render compatibility
    port = int(os.environ.get('PORT', 5000)) # Default to 5000 for local dev
    # Run Flask app (debug should be False or read from env var in production)
    is_debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    print(f"Khởi chạy ứng dụng Flask trên port {port} (Debug: {is_debug})...")
    # For production via gunicorn, this app.run() is less relevant,
    # but useful for local `python app.py` execution.
    # Gunicorn will bind to 0.0.0.0:PORT automatically.
    app.run(debug=is_debug, host='0.0.0.0', port=port)