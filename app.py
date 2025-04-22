from flask import Flask, request, render_template, redirect, url_for, session, flash
import numpy as np
import pymysql
from datetime import datetime
import os
import math
import traceback
import json 
import pandas as pd 
from werkzeug.utils import secure_filename
import uuid 

app = Flask(__name__)
app.secret_key = os.urandom(24)
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
        return None

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
    if not file_storage.filename.lower().endswith(('.xlsx', '.xls')):
        return None, "Định dạng file không hợp lệ. Chỉ chấp nhận .xlsx hoặc .xls."

    try:
        # Read only the first sheet
        df = pd.read_excel(file_storage, header=None) # Read without assuming headers

        # Attempt to find the start of the numeric matrix
        start_row, start_col = -1, -1
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                if pd.api.types.is_numeric_dtype(type(df.iloc[r, c])):
                    # Check if this looks like the top-left '1'
                    if abs(float(df.iloc[r, c]) - 1.0) < 1e-6:
                        # Look ahead to see if it's likely a matrix
                        if r + 1 < df.shape[0] and c + 1 < df.shape[1] and \
                           pd.api.types.is_numeric_dtype(type(df.iloc[r+1, c+1])):
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
            return None, "Ma trận chứa các giá trị không phải số. Vui lòng kiểm tra file Excel."

        # --- Basic Matrix Validation ---
        if matrix_np.shape != (expected_size, expected_size):
            return None, f"Kích thước ma trận không đúng. Cần {expected_size}x{expected_size}, tìm thấy {matrix_np.shape}."

        if not np.allclose(np.diag(matrix_np), 1.0):
            return None, "Đường chéo chính của ma trận phải bằng 1."

        for i in range(expected_size):
            for j in range(i + 1, expected_size):
                if abs(matrix_np[i, j] * matrix_np[j, i] - 1.0) > 1e-6 or matrix_np[i,j] <= 0 or matrix_np[j,i] <= 0:
                    return None, f"Giá trị nghịch đảo không chính xác hoặc không dương tại vị trí ({i+1},{j+1}) và ({j+1},{i+1}). Giá trị phải dương và A[j,i] = 1/A[i,j]."

        # Optional: Validate names if provided (helps catch wrong file)
        # if item_names_for_validation:
        #     # Try to read names from row/column before the matrix
        #     # This part is complex and error-prone, maybe skip for now
        #     pass

        return matrix_np, None # Success

    except Exception as e:
        traceback.print_exc()
        return None, f"Lỗi khi đọc file Excel: {e}"

def ahp_weighting(matrix):
    """Calculates weights, lambda_max, CI, CR using eigenvector method."""
    if matrix is None:
        return None, None, None, None, None

    n = matrix.shape[0]
    if n <= 0:
        return np.array([]), 0, 0, 0, 0

    if np.any(matrix <= 0):
         # This check might be redundant if parse_excel_matrix or compute_pairwise_matrix works correctly
         flash("Ma trận chứa giá trị không dương.", "error")
         return None, None, None, None, None

    try:
        eigvals, eigvecs = np.linalg.eig(matrix)
        real_eigvals = np.real(eigvals)
        lambda_max = np.max(real_eigvals) # Get the largest real eigenvalue

        # Find the eigenvector corresponding to lambda_max
        max_eigval_idx = np.argmax(real_eigvals)
        principal_eigvec = np.real(eigvecs[:, max_eigval_idx])

        # Normalize the eigenvector to get weights
        if np.all(principal_eigvec <= 1e-9): # Handle all-negative case
             principal_eigvec = -principal_eigvec
        elif np.any(principal_eigvec < -1e-9):
             # This often indicates high inconsistency. Proceed but warn.
             flash("Cảnh báo: Vector trọng số có giá trị âm, có thể do tính không nhất quán cao.", "warning")

        # Ensure weights are non-negative and sum to 1
        weights = np.maximum(0, principal_eigvec) # Set negative values to 0
        sum_weights = np.sum(weights)

        if abs(sum_weights) < 1e-9:
             flash("Lỗi: Tổng vector trọng số gần bằng 0. Không thể chuẩn hóa.", "error")
             # This could happen if matrix is degenerate or very inconsistent
             # Fallback: Geometric Mean Method (optional, adds complexity)
             # Or just return error
             return None, None, None, None, None
        else:
             weights /= sum_weights # Normalize

        # Recalculate lambda_max for consistency check based on normalized weights
        # This provides a slightly different lambda_max than np.linalg.eig sometimes,
        # but is often used in AHP for CR calculation.
        # lambda_max_check = np.sum(np.dot(matrix, weights) / weights) / n # Can cause div by zero if weights are 0
        # Use the eigenvalue directly for simplicity here
        lambda_max_for_ci = np.max(real_eigvals)

        # Calculate Consistency
        if n > 2:
            CI = (lambda_max_for_ci - n) / (n - 1)
            # Force CI >= 0 (small negative values can occur due to floating point errors)
            CI = max(0.0, CI)
            RI = RI_DICT.get(n)
            if RI is None:
                 flash(f"Lỗi: Không tìm thấy Chỉ số ngẫu nhiên (RI) cho ma trận kích thước n={n}.", "error")
                 RI = RI_DICT[max(RI_DICT.keys())] # Use max known RI as fallback, but it's wrong
                 CR = float('inf')
            elif RI == 0:
                 flash(f"Cảnh báo: Chỉ số ngẫu nhiên (RI) bằng 0 cho n={n}. CR sẽ là vô cực nếu CI > 0.", "warning")
                 CR = float('inf') if CI > 1e-9 else 0.0
            else:
                 CR = CI / RI
        elif n <= 2: # Perfectly consistent
            CI = 0.0
            RI = 0.00
            CR = 0.0
        else: # n=0, should not happen if input validation is correct
             CI = None
             RI = None
             CR = None

        # Final check for NaN/Inf in results
        if any(x is not None and (math.isnan(x) or math.isinf(x)) for x in [lambda_max_for_ci, CI, CR]) or \
           any(np.isnan(weights)) or any(np.isinf(weights)):
            flash("Lỗi: Kết quả tính toán AHP chứa NaN hoặc vô cực.", "error")
            print(f"NaN/Inf detected: lambda_max={lambda_max_for_ci}, CI={CI}, CR={CR}, weights={weights}")
            return None, None, None, None, None

        return weights, round(lambda_max_for_ci, 5), round(CI, 5), round(CR, 5), RI

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
        all_db_alternatives = False # Default to false unless DB mode succeeds

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

            conn = get_connection()
            if not conn:
                flash("Lỗi kết nối Database.", "error")
                return redirect(url_for('select_alternatives'))
            try:
                with conn.cursor() as cursor:
                    format_strings = ','.join(['%s'] * len(selected_ids))
                    # Use FIELD to preserve selection order
                    query = f"SELECT id, ten_phuong_an FROM phuong_an WHERE id IN ({format_strings}) ORDER BY FIELD(id, {format_strings})"
                    cursor.execute(query, tuple(selected_ids) * 2)
                    alternatives_from_db = cursor.fetchall()
                    if len(alternatives_from_db) != len(selected_ids):
                         flash("Không thể truy xuất tất cả phương án đã chọn.", "error")
                         return redirect(url_for('select_alternatives'))
                    alternatives = alternatives_from_db
                    all_db_alternatives = True # Success!
            except pymysql.Error as e:
                 flash(f"Lỗi truy xuất phương án từ DB: {e}", "error")
                 return redirect(url_for('select_alternatives'))
            finally:
                 if conn: conn.close()

        elif selection_mode == 'custom':
            custom_names = request.form.getlist('custom_alternative_names')
            # Filter out empty names and duplicates, preserving order
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
            # Assign None ID to custom alternatives
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
    all_alternatives_db = []
    conn = get_connection()
    db_error = None
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, ten_phuong_an FROM phuong_an ORDER BY id")
                all_alternatives_db = cursor.fetchall()
        except pymysql.Error as e:
             db_error = f"Lỗi lấy danh sách phương án từ DB: {e}"
             flash(db_error, "error")
        finally:
             if conn: conn.close()
    else:
        db_error = "Không thể kết nối đến cơ sở dữ liệu để lấy danh sách phương án."
        flash(db_error, "error")


    return render_template("select_alternatives.html",
                           all_alternatives_db=all_alternatives_db,
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

            conn = get_connection()
            if not conn:
                flash("Lỗi kết nối Database.", "error")
                return redirect(url_for('select_criteria'))
            try:
                with conn.cursor() as cursor:
                    format_strings = ','.join(['%s'] * len(selected_ids))
                    query = f"SELECT id, ten_tieu_chi FROM tieu_chi WHERE id IN ({format_strings}) ORDER BY FIELD(id, {format_strings})"
                    cursor.execute(query, tuple(selected_ids) * 2)
                    criteria_from_db = cursor.fetchall()
                    if len(criteria_from_db) != len(selected_ids):
                         flash("Không thể truy xuất tất cả tiêu chí đã chọn.", "error")
                         return redirect(url_for('select_criteria'))
                    criteria = criteria_from_db
                    all_db_criteria = True # Success!
            except pymysql.Error as e:
                 flash(f"Lỗi truy xuất tiêu chí từ DB: {e}", "error")
                 return redirect(url_for('select_criteria'))
            finally:
                 if conn: conn.close()

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

        # Clear data from subsequent steps before proceeding
        session.pop('criteria_comparison_done', None)
        session.pop('alternative_comparisons_done', None)
        session.pop('current_alt_criterion_index', None)
        # ... clear all other comparison/result related keys ...
        clear_ahp_session_data()

        session['selected_criteria'] = criteria
        session['all_db_criteria'] = all_db_criteria
        session['criteria_selected'] = True
        session.modified = True
        return redirect(url_for('compare_criteria'))

    # --- GET request ---
    all_criteria_db = []
    conn = get_connection()
    db_error = None
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, ten_tieu_chi FROM tieu_chi ORDER BY id")
                all_criteria_db = cursor.fetchall()
        except pymysql.Error as e:
             db_error = f"Lỗi lấy danh sách tiêu chí từ DB: {e}"
             flash(db_error, "error")
        finally:
             if conn: conn.close()
    else:
        db_error = "Không thể kết nối đến cơ sở dữ liệu để lấy danh sách tiêu chí."
        flash(db_error, "error")

    return render_template("select_criteria.html",
                           all_criteria_db=all_criteria_db,
                           db_error=db_error,
                           selected_alternatives=session.get('session_alternatives', []))


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
                filename = secure_filename(file.filename)
                # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # Saving might not be necessary
                # file.save(file_path)
                crit_matrix, error_msg = parse_excel_matrix(file, num_criteria, criteria_names)
                if error_msg:
                    flash(f"Lỗi xử lý file Excel tiêu chí: {error_msg}", "error")
                    # Don't proceed with calculation, redirect back to show error
                    session.pop('form_data_crit', None) # Clear any potential form data
                    session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
                    session.modified = True
                    return redirect(url_for('compare_criteria'))
                elif crit_matrix is not None:
                     input_method = "excel"
                     flash("Đã nhập ma trận so sánh tiêu chí từ file Excel.", "info")
                # else: file was empty, proceed to check form

        # If no valid matrix from Excel, try form data
        if crit_matrix is None:
            input_method = "form"
            crit_matrix = compute_pairwise_matrix("pc", criteria_names, request.form)
            if crit_matrix is None:
                # compute_pairwise_matrix already flashed the error
                session['form_data_crit'] = request.form # Keep form data for repopulation
                session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
                session.modified = True
                return redirect(url_for('compare_criteria'))

        # --- Matrix obtained (from form or Excel), proceed with AHP calculation ---
        crit_weights, crit_lambda_max, crit_ci, crit_cr, crit_ri = ahp_weighting(crit_matrix)

        if crit_weights is None:
            # ahp_weighting flashed the error
            if input_method == "form":
                 session['form_data_crit'] = request.form # Keep form data if it came from form
            session.pop('crit_lambda_max', None); session.pop('crit_ci', None); session.pop('crit_cr', None); session.pop('crit_ri', None)
            session.modified = True
            return redirect(url_for('compare_criteria'))

        # Store results in session temporarily (for display if CR fails) and permanently if CR passes
        session['crit_matrix'] = crit_matrix.tolist() # Always store matrix for potential display
        session['crit_lambda_max'] = crit_lambda_max
        session['crit_ci'] = crit_ci
        session['crit_cr'] = crit_cr
        session['crit_ri'] = crit_ri

        if crit_cr > CR_THRESHOLD:
            flash(f"Tỷ số nhất quán (CR = {crit_cr:.4f}) vượt ngưỡng ({CR_THRESHOLD:.2f}). Vui lòng xem lại các so sánh tiêu chí.", "error")
            session['criteria_comparison_done'] = False # Mark as not done
            if input_method == "form":
                 session['form_data_crit'] = request.form # Keep form data
            else:
                 session.pop('form_data_crit', None) # Clear form data if input was Excel
            session.modified = True
            # Redirect back to the comparison page, results will be shown from session
            return redirect(url_for('compare_criteria'))
        else:
            # --- CR is acceptable ---
            flash(f"So sánh tiêu chí thành công (CR = {crit_cr:.4f}). Tiếp tục so sánh phương án.", "success")
            session['crit_weights'] = crit_weights.tolist() # Store weights permanently
            session['criteria_comparison_done'] = True
            session.pop('form_data_crit', None) # Clear form data on success

            # Initialize structures for alternative comparisons
            num_alternatives = len(session.get('session_alternatives', []))
            session['alt_matrices_all'] = [None] * num_criteria
            session['alt_weights_all'] = [None] * num_criteria
            session['alt_lambda_max_all'] = [None] * num_criteria
            session['alt_ci_all'] = [None] * num_criteria
            session['alt_cr_all'] = [None] * num_criteria
            session['alt_ri_all'] = [None] * num_criteria
            session['current_alt_criterion_index'] = 0 # Start with the first criterion
            clear_temporary_alt_data(num_criteria) # Clear any leftover temp data
            session.modified = True

            return redirect(url_for('compare_alternatives'))

    # --- GET request ---
    # Retrieve potentially stored form data or calculation results for display
    form_data = session.get('form_data_crit', None)
    crit_lambda_max = session.get('crit_lambda_max')
    crit_ci = session.get('crit_ci')
    crit_cr = session.get('crit_cr')
    crit_ri = session.get('crit_ri')
    # Don't clear form_data here, it should persist until successful POST

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
    if not selected_criteria: # Should be caught earlier, but double-check
         flash("Không có tiêu chí nào được chọn.", "error")
         return redirect(url_for('select_criteria'))


    alternative_names = [a['ten_phuong_an'] for a in alternatives]
    num_alternatives = len(alternatives)
    num_criteria = len(selected_criteria)
    current_index = session.get('current_alt_criterion_index', 0)

    # Check if all comparisons are done
    if current_index >= num_criteria:
        session['alternative_comparisons_done'] = True
        session.modified = True
        flash("Tất cả so sánh phương án đã hoàn thành.", "info")
        return redirect(url_for('calculate_results')) # Go to results

    current_criterion = selected_criteria[current_index]
    alt_matrix = None
    input_method = "form"

    if request.method == "POST":
         # Ensure the post request corresponds to the current criterion index
         # (This helps prevent issues if user navigates back/forward weirdly)
         try:
              posted_criterion_id = int(request.form.get('criterion_id_hidden', -1))
              if posted_criterion_id != current_criterion.get('id', -2): # Use ID if available, else check name? ID is better. Or just trust the session index.
                   # For simplicity, we'll rely on the session index being correct.
                   # If needed, add hidden fields to track the intended criterion.
                   pass
         except (ValueError, TypeError):
              flash("Lỗi xử lý ID tiêu chí ẩn.", "warning")
              # Decide how to handle: proceed cautiously or redirect? Let's proceed.


         # Check for Excel file upload
         if f'alternative_excel_file_{current_index}' in request.files:
             file = request.files[f'alternative_excel_file_{current_index}']
             if file and file.filename != '':
                filename = secure_filename(file.filename)
                # file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"alt_{current_criterion.get('id', current_index)}_{filename}")
                # file.save(file_path)
                alt_matrix, error_msg = parse_excel_matrix(file, num_alternatives, alternative_names)
                if error_msg:
                    flash(f"Lỗi xử lý file Excel cho tiêu chí '{current_criterion['ten_tieu_chi']}': {error_msg}", "error")
                    session.pop('form_data_alt', None)
                    session.pop(f'temp_alt_lambda_max_{current_index}', None); session.pop(f'temp_alt_ci_{current_index}', None); session.pop(f'temp_alt_cr_{current_index}', None); session.pop(f'temp_alt_ri_{current_index}', None)
                    session.modified = True
                    return redirect(url_for('compare_alternatives')) # Redirect back to same criterion
                elif alt_matrix is not None:
                     input_method = "excel"
                     flash(f"Đã nhập ma trận so sánh phương án cho '{current_criterion['ten_tieu_chi']}' từ file Excel.", "info")
                 # else: file was empty, check form

         # If no valid matrix from Excel, try form data
         if alt_matrix is None:
            input_method = "form"
            # Prefix uses index to avoid collisions if IDs are None or duplicated
            prefix = f"alt_pc_{current_index}"
            alt_matrix = compute_pairwise_matrix(prefix, alternative_names, request.form)

            if alt_matrix is None:
                # compute_pairwise_matrix flashed the error
                session['form_data_alt'] = request.form # Store form data for repopulation
                session.pop(f'temp_alt_lambda_max_{current_index}', None); session.pop(f'temp_alt_ci_{current_index}', None); session.pop(f'temp_alt_cr_{current_index}', None); session.pop(f'temp_alt_ri_{current_index}', None)
                session.modified = True
                return redirect(url_for('compare_alternatives')) # Redirect back to same criterion

         # --- Matrix obtained, perform AHP calculation ---
         alt_weights, alt_lambda_max, alt_ci, alt_cr, alt_ri = ahp_weighting(alt_matrix)

         if alt_weights is None:
             # ahp_weighting flashed error
             if input_method == "form":
                  session['form_data_alt'] = request.form
             session.pop(f'temp_alt_lambda_max_{current_index}', None); session.pop(f'temp_alt_ci_{current_index}', None); session.pop(f'temp_alt_cr_{current_index}', None); session.pop(f'temp_alt_ri_{current_index}', None)
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
             session['alternative_comparisons_done'] = False # Mark as not fully done
             if input_method == "form":
                  session['form_data_alt'] = request.form
             else:
                  session.pop('form_data_alt', None)
             session.modified = True
             # Redirect back to the same criterion page for correction
             return redirect(url_for('compare_alternatives'))
         else:
             # --- Consistent! Store results permanently for this criterion ---
             flash(f"So sánh phương án theo '{current_criterion['ten_tieu_chi']}' đã lưu (CR = {alt_cr:.4f}).", "success")

             # Ensure lists exist and have correct length before assigning
             def ensure_session_list(key, length, default_val=None):
                 if key not in session or not isinstance(session[key], list) or len(session[key]) != length:
                     session[key] = [default_val] * length
                 # Handle case where list exists but wrong length (e.g., back navigation)
                 elif len(session[key]) != length:
                      session[key] = [default_val] * length
                 return session[key]

             alt_matrices_all = ensure_session_list('alt_matrices_all', num_criteria, default_val=[])
             alt_weights_all = ensure_session_list('alt_weights_all', num_criteria)
             alt_lambda_max_all = ensure_session_list('alt_lambda_max_all', num_criteria)
             alt_ci_all = ensure_session_list('alt_ci_all', num_criteria)
             alt_cr_all = ensure_session_list('alt_cr_all', num_criteria)
             alt_ri_all = ensure_session_list('alt_ri_all', num_criteria)

             # Store permanent results
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

             # Clear temporary data for this index and general form data
             clear_temporary_alt_data_for_index(current_index)
             session.pop('form_data_alt', None)

             # Move to the next criterion
             next_index = current_index + 1
             session['current_alt_criterion_index'] = next_index
             session.modified = True

             if next_index >= num_criteria:
                 session['alternative_comparisons_done'] = True
                 return redirect(url_for('calculate_results')) # All done, go to results
             else:
                 # Important: Redirect to GET to load the next criterion's page cleanly
                 return redirect(url_for('compare_alternatives'))

    # --- GET request ---
    # Retrieve temporary data if a previous POST for this index failed CR check
    form_data = session.get('form_data_alt', None) # General form data (only if last fail was form)
    # Temp results are specific to the index
    alt_lambda_max = session.get(f'temp_alt_lambda_max_{current_index}')
    alt_ci = session.get(f'temp_alt_ci_{current_index}')
    alt_cr = session.get(f'temp_alt_cr_{current_index}')
    alt_ri = session.get(f'temp_alt_ri_{current_index}')

    # Clear the general form data after retrieving it for display
    # It should only be used once for repopulation on the GET after a form-based failure
    if 'form_data_alt' in session:
         session.pop('form_data_alt')
         session.modified = True

    return render_template("compare_alternatives.html",
                           criterion=current_criterion,
                           alternatives=alternatives,
                           alternative_names=alternative_names,
                           form_data=form_data, # Pass potentially repopulated form data
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

    # Check if alternative comparisons are actually finished
    num_criteria = len(session.get('selected_criteria', []))
    num_alternatives = len(session.get('session_alternatives', []))
    current_alt_index = session.get('current_alt_criterion_index', -1)

    if not session.get('alternative_comparisons_done'):
        # Double check if the index implies completion
        if current_alt_index == num_criteria and num_criteria > 0:
             session['alternative_comparisons_done'] = True
             session.modified = True
             # Proceed
        else:
             flash(f"So sánh phương án chưa hoàn thành (đang ở tiêu chí {current_alt_index+1}/{num_criteria}).", "warning")
             return redirect(url_for('compare_alternatives'))

    # Validate presence and basic structure of required session data
    required_keys = [
        'crit_weights', 'alt_weights_all', 'session_alternatives', 'selected_criteria',
        'crit_matrix', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri',
        'alt_matrices_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all',
        'all_db_alternatives', 'all_db_criteria' # Need these flags for saving logic
    ]
    missing_or_invalid = []
    for key in required_keys:
        data = session.get(key)
        if data is None:
            missing_or_invalid.append(f"'{key}' is missing")
            continue
        # Check list lengths for alternative-related data
        if key.startswith('alt_') and key.endswith('_all'):
             if not isinstance(data, list) or len(data) != num_criteria:
                 missing_or_invalid.append(f"'{key}' has incorrect length (found {len(data) if isinstance(data, list) else 'N/A'}, expected {num_criteria})")
             elif any(x is None for x in data):
                  missing_indices = [i for i, x in enumerate(data) if x is None]
                  missing_or_invalid.append(f"'{key}' has missing data at indices: {missing_indices}")

    if missing_or_invalid:
        error_message = "Dữ liệu session không đầy đủ/hợp lệ để tính kết quả: " + "; ".join(missing_or_invalid) + ". Vui lòng thử lại."
        flash(error_message, "error")
        # Try to redirect intelligently based on missing keys
        if any('alt_' in s for s in missing_or_invalid): return redirect(url_for('compare_alternatives'))
        elif any('crit_' in s for s in missing_or_invalid): return redirect(url_for('compare_criteria'))
        else: return redirect(url_for('clear_session_and_start')) # Fallback

    # --- Perform Final Calculation ---
    try:
        crit_weights = np.array(session['crit_weights'])
        # alt_weights_all should be a list of lists/arrays
        alt_weights_all_list = session['alt_weights_all']

        # Convert list of lists/arrays into a 2D numpy array
        # Rows = Criteria, Columns = Alternatives
        alt_weights_matrix = np.array(alt_weights_all_list).T # Transpose needed here!

        # --- Shape Validation ---
        if crit_weights.shape != (num_criteria,):
            raise ValueError(f"Kích thước trọng số tiêu chí không đúng ({crit_weights.shape}), cần ({num_criteria},)")
        # After transpose, alt_weights_matrix should be num_alternatives x num_criteria
        if alt_weights_matrix.shape != (num_alternatives, num_criteria):
             raise ValueError(f"Kích thước ma trận trọng số PA sau chuyển vị không đúng ({alt_weights_matrix.shape}), cần ({num_alternatives}, {num_criteria})")

        # --- Value Validation ---
        if np.isnan(crit_weights).any() or np.isinf(crit_weights).any(): raise ValueError("NaN/Inf trong trọng số tiêu chí.")
        if np.isnan(alt_weights_matrix).any() or np.isinf(alt_weights_matrix).any(): raise ValueError("NaN/Inf trong ma trận trọng số phương án.")
        if abs(np.sum(crit_weights) - 1.0) > 1e-5: flash(f"Cảnh báo: Tổng trọng số tiêu chí ~ {np.sum(crit_weights):.6f} (nên bằng 1).", "warning")
        # Check sums of local weights (columns of transposed matrix, i.e., rows of original)
        for i in range(num_criteria):
             col_sum = np.sum(alt_weights_matrix[:, i]) # Sum down columns (weights for crit i)
             if abs(col_sum - 1.0) > 1e-5:
                 crit_name = session['selected_criteria'][i]['ten_tieu_chi']
                 flash(f"Cảnh báo: Tổng trọng số PA cho tiêu chí '{crit_name}' ~ {col_sum:.6f} (nên bằng 1).", "warning")

        # --- Final Score Calculation ---
        # final_scores = alt_weights_matrix @ crit_weights # Matrix multiplication
        final_scores_vector = np.dot(alt_weights_matrix, crit_weights) # Equivalent

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
                    'id': alt.get('id'), # Will be None for custom
                    'name': alt_name,
                    'score': score,
                    'is_best': is_best
                }
                results_display.append(display_item)
                if is_best:
                    best_alternative_info = display_item # Store best info separately

            # Sort for display ranking
            results_display.sort(key=lambda x: x['score'], reverse=True)

        session['final_scores'] = final_scores_dict # Store for potential later use
        session['best_alternative_info'] = best_alternative_info
        session.modified = True

    except (ValueError, TypeError, IndexError) as e:
         flash(f"Lỗi trong quá trình tính toán cuối cùng: {e}", "error")
         print(f"Final Calculation Error Details: {e}")
         traceback.print_exc()
         # Provide intermediate data to results template for debugging if possible
         intermediate_results = get_intermediate_results_for_display()
         return render_template("results.html", error=f"Lỗi tính toán: {e}", intermediate=intermediate_results)
    except Exception as e:
         flash(f"Đã xảy ra lỗi không mong muốn trong quá trình tính toán cuối cùng: {e}.", "error")
         print(f"Unexpected Final Calculation Error: {e}")
         traceback.print_exc()
         return render_template("error.html", message=f"Lỗi không mong muốn: {e}")


    # --- Save results to database (ONLY if using DB alternatives AND DB criteria) ---
    can_save_to_db = session.get('all_db_alternatives', False) and session.get('all_db_criteria', False)
    save_attempted = False
    save_successful = False

    if can_save_to_db and results_display:
        save_attempted = True
        conn = get_connection()
        if conn:
            analysis_group_id = str(uuid.uuid4()) # Generate unique ID for this analysis run
            timestamp = datetime.now()
            try:
                with conn.cursor() as cursor:
                    conn.begin() # Start transaction

                    # 1. Insert rows for final alternative scores
                    insert_alt_query = """
                        INSERT INTO ket_qua (analysis_group_id, thoi_gian, phuong_an_id, phuong_an_ten,
                                             tieu_chi_id, tieu_chi_ten, is_alternative, is_db_source,
                                             final_score, is_best, criterion_weight)
                        VALUES (%s, %s, %s, %s, NULL, '', TRUE, TRUE, %s, %s, NULL)
                    """
                    alt_values_to_insert = []
                    for result in results_display:
                         # Only save DB alternatives (ID is not None)
                         if result['id'] is not None:
                             alt_values_to_insert.append((
                                 analysis_group_id, timestamp, result['id'], result['name'],
                                 result['score'], result['is_best']
                             ))
                    if alt_values_to_insert:
                        cursor.executemany(insert_alt_query, alt_values_to_insert)

                    # 2. Insert rows for criteria weights used in this analysis
                    insert_crit_query = """
                        INSERT INTO ket_qua (analysis_group_id, thoi_gian, phuong_an_id, phuong_an_ten,
                                             tieu_chi_id, tieu_chi_ten, is_alternative, is_db_source,
                                             final_score, is_best, criterion_weight)
                        VALUES (%s, %s, NULL, '', %s, %s, FALSE, TRUE, NULL, NULL, %s)
                    """
                    crit_values_to_insert = []
                    db_criteria = session['selected_criteria'] # Should only contain DB criteria if can_save_to_db is True
                    db_crit_weights = session['crit_weights'] # Corresponding weights
                    for i, crit in enumerate(db_criteria):
                        # Only save DB criteria (ID is not None) -redundant check due to can_save flag, but safe
                        if crit.get('id') is not None:
                             crit_values_to_insert.append((
                                 analysis_group_id, timestamp, crit['id'], crit['ten_tieu_chi'],
                                 db_crit_weights[i]
                             ))
                    if crit_values_to_insert:
                        cursor.executemany(insert_crit_query, crit_values_to_insert)

                    # If both inserts had data and executed without error
                    if alt_values_to_insert and crit_values_to_insert:
                        conn.commit()
                        save_successful = True
                        flash("Kết quả phân tích đã được lưu vào cơ sở dữ liệu.", "success")
                    else:
                         conn.rollback() # Rollback if one part was empty
                         # Should not happen if can_save_to_db logic is correct, but handle defensively
                         flash("Lỗi logic: Không có đủ dữ liệu DB để lưu mặc dù cờ cho phép.", "warning")

            except pymysql.Error as e:
                conn.rollback()
                flash(f"Lỗi lưu kết quả vào cơ sở dữ liệu: {e}", "error")
                print(f"DB Save Error: {e}")
                traceback.print_exc()
            except Exception as e:
                conn.rollback()
                flash(f"Đã xảy ra lỗi không mong muốn khi lưu kết quả: {e}", "error")
                print(f"Unexpected DB Save Error: {e}")
                traceback.print_exc()
            finally:
                if conn: conn.close()
        else:
            flash("Không thể kết nối đến cơ sở dữ liệu để lưu kết quả.", "error")
            save_attempted = True # Attempted connection but failed
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
    conn = get_connection()
    grouped_history = {} # Key: analysis_group_id
    db_error = None

    if conn:
        try:
            with conn.cursor() as cursor:
                # Get recent unique analysis groups
                cursor.execute("""
                    SELECT DISTINCT analysis_group_id, MAX(thoi_gian) as analysis_time
                    FROM ket_qua
                    WHERE is_db_source = TRUE
                    GROUP BY analysis_group_id
                    ORDER BY analysis_time DESC
                    LIMIT 20
                """)
                analysis_groups = cursor.fetchall()

                for group in analysis_groups:
                    group_id = group['analysis_group_id']
                    group_time = group['analysis_time']
                    group_data = {
                        'timestamp_obj': group_time,
                        'timestamp_str': group_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'group_id': group_id,
                        'alternatives': [],
                        'criteria': []
                    }

                    # Get alternative results for this group
                    cursor.execute("""
                        SELECT phuong_an_ten, final_score, is_best
                        FROM ket_qua
                        WHERE analysis_group_id = %s AND is_alternative = TRUE AND is_db_source = TRUE
                        ORDER BY final_score DESC
                    """, (group_id,))
                    group_data['alternatives'] = cursor.fetchall()

                    # Get criteria weights for this group
                    cursor.execute("""
                        SELECT tieu_chi_ten, criterion_weight
                        FROM ket_qua
                        WHERE analysis_group_id = %s AND is_alternative = FALSE AND is_db_source = TRUE
                        ORDER BY criterion_weight DESC
                    """, (group_id,))
                    group_data['criteria'] = cursor.fetchall()

                    if group_data['alternatives'] or group_data['criteria']:
                         grouped_history[group_id] = group_data

        except pymysql.Error as e:
             db_error = f"Lỗi lấy lịch sử kết quả: {e}"
             flash(db_error, "error"); print(f"DB History Error: {e}"); traceback.print_exc()
        except Exception as e:
             db_error = f"Đã xảy ra lỗi không mong muốn khi lấy lịch sử: {e}"
             flash(db_error, "error"); print(f"Unexpected History Error: {e}"); traceback.print_exc()
        finally:
            if conn: conn.close()
    else:
        db_error = "Không thể kết nối đến cơ sở dữ liệu để lấy lịch sử."
        flash(db_error, "error")

    # Sort groups by timestamp descending before passing to template
    sorted_history_list = sorted(grouped_history.values(), key=lambda item: item['timestamp_obj'], reverse=True)

    return render_template("results_history.html", history_list=sorted_history_list, db_error=db_error)

# --- Helper Functions ---

def get_intermediate_results_for_display():
    """Safely retrieves intermediate results from session for the results page."""
    intermediate = {}
    try:
        intermediate = {
            'criteria': session.get('selected_criteria', []),
            'crit_matrix': session.get('crit_matrix'),
            'crit_weights': session.get('crit_weights'),
            'crit_lambda_max': session.get('crit_lambda_max'),
            'crit_ci': session.get('crit_ci'),
            'crit_cr': session.get('crit_cr'),
            'crit_ri': session.get('crit_ri'),
            'alternatives': session.get('session_alternatives', []),
            'alt_matrices_all': session.get('alt_matrices_all'),
            'alt_weights_all': session.get('alt_weights_all'),
            'alt_lambda_max_all': session.get('alt_lambda_max_all'),
            'alt_ci_all': session.get('alt_ci_all'),
            'alt_cr_all': session.get('alt_cr_all'),
            'alt_ri_all': session.get('alt_ri_all'),
        }
        # Basic validation of lengths
        num_crit_check = len(intermediate.get('criteria', []))
        if len(intermediate.get('alt_matrices_all', [])) != num_crit_check:
             flash("Cảnh báo hiển thị: Số lượng ma trận phương án không khớp số tiêu chí.", "warning")
        if len(intermediate.get('alt_weights_all', [])) != num_crit_check:
             flash("Cảnh báo hiển thị: Số lượng trọng số phương án không khớp số tiêu chí.", "warning")
    except Exception as e:
         flash(f"Lỗi khi chuẩn bị dữ liệu trung gian để hiển thị: {e}", "warning")
         print(f"Error preparing intermediate results: {e}")
         traceback.print_exc()
         intermediate = {} # Reset on error
    return intermediate

def clear_temporary_alt_data_for_index(index):
     """Clears temporary session keys for a specific alt comparison index."""
     session.pop(f'temp_alt_matrix_{index}', None)
     session.pop(f'temp_alt_lambda_max_{index}', None)
     session.pop(f'temp_alt_ci_{index}', None)
     session.pop(f'temp_alt_cr_{index}', None)
     session.pop(f'temp_alt_ri_{index}', None)

def clear_temporary_alt_data(num_criteria):
     """Clears all temporary alt comparison keys."""
     # Estimate max criteria if num_criteria is unreliable (e.g., 0)
     max_crit_guess = max(num_criteria if isinstance(num_criteria, int) and num_criteria > 0 else 0, 20)
     for i in range(max_crit_guess):
         clear_temporary_alt_data_for_index(i)
     session.pop('form_data_alt', None) # Clear general alt form data too

def clear_ahp_session_data():
    """Clears session keys related to AHP steps (criteria onwards)."""
    keys_to_clear = [
        # Criteria Selection & Comparison
        'selected_criteria', 'all_db_criteria', 'criteria_selected',
        'crit_matrix', 'crit_weights', 'crit_lambda_max', 'crit_ci', 'crit_cr', 'crit_ri',
        'criteria_comparison_done', 'form_data_crit',
        # Alternative Comparison (permanent and temporary)
        'alt_matrices_all', 'alt_weights_all', 'alt_lambda_max_all', 'alt_ci_all', 'alt_cr_all', 'alt_ri_all',
        'current_alt_criterion_index', 'alternative_comparisons_done', 'form_data_alt',
        # Results
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
    # Also clear step 0 data
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

# --- Error Handlers ---
@app.errorhandler(404)
def page_not_found(e):
     flash("Trang yêu cầu không được tìm thấy (404).", "error")
     return render_template('error.html', message='Trang không tìm thấy (404)'), 404

@app.errorhandler(500)
def internal_server_error(e):
     print(f"Internal Server Error: {e}")
     traceback.print_exc()
     flash("Đã xảy ra lỗi máy chủ nội bộ (500). Vui lòng thử lại.", "error")
     # Log the error properly in a real application
     return render_template('error.html', message='Lỗi Máy chủ Nội bộ (500)'), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    flash("File tải lên quá lớn. Vui lòng chọn file nhỏ hơn (giới hạn 1MB).", "error")
    # Redirect back to the page where the upload likely happened
    # This is tricky, maybe just redirect to a sensible default like compare_criteria
    if 'current_alt_criterion_index' in session:
        return redirect(url_for('compare_alternatives'))
    elif 'criteria_selected' in session:
        return redirect(url_for('compare_criteria'))
    else:
        return redirect(url_for('select_criteria'))


# --- Main Execution ---
if __name__ == "__main__":
    # Test DB connection on startup
    print("Kiểm tra kết nối cơ sở dữ liệu...")
    conn_test = get_connection()
    if conn_test is None:
         print("\n*** CẢNH BÁO: Không thể kết nối đến cơ sở dữ liệu! ***")
         print("Vui lòng kiểm tra thông tin đăng nhập (host, user, password, database)")
         print("trong hàm get_connection() và đảm bảo MySQL server đang chạy.")
         print("Chạy lệnh SQL để tạo bảng và dữ liệu mặc định nếu chưa có.\n")
    else:
        print("Kết nối cơ sở dữ liệu thành công.")
        conn_test.close()
    # Run the Flask development server
    print("Khởi chạy ứng dụng Flask...")
    app.run(debug=True) # Set debug=False for production

