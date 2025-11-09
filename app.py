from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- Config / paths ---
CSV_PATH = "expected_ctc_processed.csv"
MODEL_PATH = "model.pkl"

# Allowed ranges
GRAD_YEAR_MIN = 1990
GRAD_YEAR_MAX = 2025
MAX_EXPERIENCE = 60

# --- Load model and CSV (required) ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place model.pkl in the project folder.")
model = pickle.load(open(MODEL_PATH, "rb"))

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Place expected_ctc_processed.csv in the project folder.")
df_full = pd.read_csv(CSV_PATH)

# Drop identifier and target columns (if present)
drop_cols = [c for c in ["IDX", "Applicant_ID", "Expected_CTC"] if c in df_full.columns]
X_df = df_full.drop(columns=drop_cols, errors="ignore")

# The exact columns generated during training using pd.get_dummies(X_df)
training_dummies = pd.get_dummies(X_df)
TRAINING_COLUMNS = training_dummies.columns.tolist()

# Detect categorical and numeric columns (for dynamic form generation)
CATEGORICAL_COLS = X_df.select_dtypes(include='object').columns.tolist()
NUMERIC_COLS = X_df.select_dtypes(include=[np.number]).columns.tolist()

# Unique options for each categorical col
DROPDOWN_OPTIONS = {}
for col in CATEGORICAL_COLS:
    vals = X_df[col].dropna().unique().tolist()
    try:
        DROPDOWN_OPTIONS[col] = sorted(vals)
    except Exception:
        DROPDOWN_OPTIONS[col] = vals

# Helper: parse numeric safely
def parse_numeric(form, name):
    raw = form.get(name)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except:
        return None

# Validation returns tuple (error_list, field_errors_dict)
def validate_inputs(form):
    errors = []
    field_errors = {}

    # Required numeric fields to check presence
    required = ['Total_Experience', 'Total_Experience_in_field_applied', 'Current_CTC',
                'No_Of_Companies_worked', 'Certifications']
    for r in required:
        if not form.get(r) or str(form.get(r)).strip() == "":
            msg = "Required."
            errors.append(f"{r.replace('_',' ')} is required.")
            field_errors[r] = msg

    # parse numeric
    total_exp = parse_numeric(form, 'Total_Experience')
    field_exp = parse_numeric(form, 'Total_Experience_in_field_applied')
    current_ctc = parse_numeric(form, 'Current_CTC')
    companies = parse_numeric(form, 'No_Of_Companies_worked')
    certs = parse_numeric(form, 'Certifications')
    grad_year = parse_numeric(form, 'Passing_Year_Of_Graduation')
    pg_year = parse_numeric(form, 'Passing_Year_Of_PG')

    # Range checks & field-level messages
    if total_exp is not None:
        if total_exp < 0 or total_exp > MAX_EXPERIENCE:
            errors.append(f"Total experience must be between 0 and {MAX_EXPERIENCE}.")
            field_errors['Total_Experience'] = f"Must be 0–{MAX_EXPERIENCE}."

    if field_exp is not None:
        if field_exp < 0:
            errors.append("Experience in field cannot be negative.")
            field_errors['Total_Experience_in_field_applied'] = "Cannot be negative."
        if total_exp is not None and field_exp > total_exp:
            errors.append("Experience in applied field cannot exceed total experience.")
            field_errors['Total_Experience_in_field_applied'] = "Cannot exceed total experience."

    if grad_year is not None:
        if grad_year < GRAD_YEAR_MIN or grad_year > GRAD_YEAR_MAX:
            errors.append(f"Passing year of graduation must be between {GRAD_YEAR_MIN} and {GRAD_YEAR_MAX}.")
            field_errors['Passing_Year_Of_Graduation'] = f"{GRAD_YEAR_MIN}–{GRAD_YEAR_MAX}"

    if pg_year is not None:
        if pg_year < GRAD_YEAR_MIN or pg_year > GRAD_YEAR_MAX:
            errors.append(f"Passing year of PG must be between {GRAD_YEAR_MIN} and {GRAD_YEAR_MAX}.")
            field_errors['Passing_Year_Of_PG'] = f"{GRAD_YEAR_MIN}–{GRAD_YEAR_MAX}"

    if current_ctc is not None and current_ctc < 0:
        errors.append("Current CTC cannot be negative.")
        field_errors['Current_CTC'] = "Cannot be negative."

    if companies is not None and companies < 0:
        errors.append("No. of companies worked cannot be negative.")
        field_errors['No_Of_Companies_worked'] = "Cannot be negative."

    if certs is not None and certs < 0:
        errors.append("Certifications cannot be negative.")
        field_errors['Certifications'] = "Cannot be negative."

    # Logical education consistency: PG/PHD specialization vs Education
    education = (form.get('Education') or "").strip()
    pg_spec = (form.get('PG_Specialization') or "").strip()
    phd_spec = (form.get('PHD_Specialization') or "").strip()
    if education != "":
        edu_lower = education.lower()
        if edu_lower in ['under grad', 'grad'] and pg_spec:
            msg = "PG specialization not valid for selected Education."
            errors.append(msg)
            field_errors['PG_Specialization'] = "Not valid for this Education."
        if edu_lower != 'doctorate' and phd_spec:
            msg = "PHD specialization not valid unless Education is Doctorate."
            errors.append(msg)
            field_errors['PHD_Specialization'] = "Not valid for this Education."

    # Inhand offer check
    inhand = (form.get('Inhand_Offer') or "").strip()
    if inhand and inhand not in ['Y', 'N']:
        errors.append("Inhand Offer must be 'Y' or 'N'.")
        field_errors['Inhand_Offer'] = "Must be Y or N."

    # department and role presence
    if not form.get('Department') or form.get('Department').strip() == "":
        errors.append("Please select a Department.")
        field_errors['Department'] = "Required."
    if not form.get('Role') or form.get('Role').strip() == "":
        errors.append("Please select a Role.")
        field_errors['Role'] = "Required."

    return errors, field_errors

# Build aligned input row using pd.get_dummies + reindex to TRAINING_COLUMNS
def build_input_dataframe(form):
    data = {}
    for col in NUMERIC_COLS:
        v = parse_numeric(form, col)
        data[col] = v if v is not None else np.nan
    for col in CATEGORICAL_COLS:
        v = form.get(col)
        data[col] = v if (v is not None and v != "") else np.nan
    input_df = pd.DataFrame([data])
    input_dummies = pd.get_dummies(input_df)
    aligned = input_dummies.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    return aligned

# Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html",
                           dropdowns=DROPDOWN_OPTIONS,
                           numeric_cols=NUMERIC_COLS,
                           categorical_cols=CATEGORICAL_COLS,
                           prediction_text=None,
                           error_messages=None,
                           field_errors={},
                           input_values={})

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    errors, field_errors = validate_inputs(form)
    if errors:
        return render_template("index.html",
                               dropdowns=DROPDOWN_OPTIONS,
                               numeric_cols=NUMERIC_COLS,
                               categorical_cols=CATEGORICAL_COLS,
                               prediction_text=None,
                               error_messages=errors,
                               field_errors=field_errors,
                               input_values=form)

    try:
        input_aligned = build_input_dataframe(form)
        pred = model.predict(input_aligned)[0]
        pred_int = int(round(pred))
        formatted = f"₹ {pred_int:,} per year"
        return render_template("index.html",
                               dropdowns=DROPDOWN_OPTIONS,
                               numeric_cols=NUMERIC_COLS,
                               categorical_cols=CATEGORICAL_COLS,
                               prediction_text=formatted,
                               error_messages=None,
                               field_errors={},
                               input_values=form)
    except Exception as e:
        return render_template("index.html",
                               dropdowns=DROPDOWN_OPTIONS,
                               numeric_cols=NUMERIC_COLS,
                               categorical_cols=CATEGORICAL_COLS,
                               prediction_text=None,
                               error_messages=[f"Prediction error: {e}"],
                               field_errors={},
                               input_values=form)

if __name__ == "__main__":
    app.run(debug=True)
