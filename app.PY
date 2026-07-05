"""
AI LungCare — Flask REST API (starter/reference implementation)
=================================================================
Covers: secure auth (JWT + bcrypt), patient/doctor/admin endpoints,
AI risk-scoring engine (rule-based, model-swappable), and emergency
alert logic — matching database/schema.sql.

Run:
    pip install flask flask-cors flask-jwt-extended mysql-connector-python bcrypt --break-system-packages
    export DB_PASSWORD="yourpassword"
    python app.py

This is a starter reference, not a full production build — wire in
proper config management, migrations (e.g. Alembic/Flyway), logging,
and a real ML model behind ai_engine.calculate_risk() before deployment.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity, get_jwt
)
import mysql.connector
from mysql.connector import pooling
import bcrypt
import os
from datetime import timedelta
from functools import wraps

# ------------------------------------------------------------------
# APP CONFIG
# ------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "change-this-in-production")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=8)
jwt = JWTManager(app)

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "ai_lungcare"),
}

# Connection pool — avoids opening a new MySQL connection per request
db_pool = pooling.MySQLConnectionPool(pool_name="lungcare_pool", pool_size=8, **DB_CONFIG)


def get_db():
    return db_pool.get_connection()


# ------------------------------------------------------------------
# ROLE-BASED ACCESS CONTROL DECORATOR
# ------------------------------------------------------------------
def require_role(*roles):
    def decorator(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            claims = get_jwt()
            if claims.get("role") not in roles:
                return jsonify({"error": "Forbidden: insufficient role"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ==================================================================
# 1. AUTH
# ==================================================================
@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    required = ["email", "password", "full_name", "role"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    password_hash = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt()).decode()

    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (email, password_hash, role) VALUES (%s, %s, %s)",
            (data["email"], password_hash, data["role"]),
        )
        user_id = cur.lastrowid

        if data["role"] == "patient":
            cur.execute(
                "INSERT INTO patients (user_id, full_name, age, gender, height_cm, weight_kg, phone) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (user_id, data["full_name"], data.get("age"), data.get("gender"),
                 data.get("height_cm"), data.get("weight_kg"), data.get("phone")),
            )
        elif data["role"] == "doctor":
            cur.execute(
                "INSERT INTO doctors (user_id, full_name, specialization, phone) VALUES (%s, %s, %s, %s)",
                (user_id, data["full_name"], data.get("specialization", "Pulmonology"), data.get("phone")),
            )
        conn.commit()
        return jsonify({"message": "Registered successfully", "user_id": user_id}), 201
    except mysql.connector.IntegrityError:
        conn.rollback()
        return jsonify({"error": "Email already registered"}), 409
    finally:
        cur.close()
        conn.close()


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE email = %s AND is_active = TRUE", (data.get("email"),))
    user = cur.fetchone()
    cur.close()
    conn.close()

    if not user or not bcrypt.checkpw(data.get("password", "").encode(), user["password_hash"].encode()):
        return jsonify({"error": "Invalid credentials"}), 401

    token = create_access_token(
        identity=str(user["user_id"]),
        additional_claims={"role": user["role"], "email": user["email"]},
    )
    return jsonify({"access_token": token, "role": user["role"]}), 200


# ==================================================================
# 2. PATIENT PROFILE, MEDICAL HISTORY, SYMPTOMS, CLINICAL PARAMS
# ==================================================================
@app.route("/api/patients/<int:patient_id>/history", methods=["POST"])
@require_role("patient", "doctor")
def save_medical_history(patient_id):
    d = request.get_json(force=True)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO medical_history
           (patient_id, smoking_status, pack_years, passive_smoking, occupation,
            dust_exposure, chemical_exposure, has_asthma, has_tuberculosis,
            family_history_copd, current_medications, alcohol_consumption, exercise_habits)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (patient_id, d.get("smoking_status"), d.get("pack_years"), d.get("passive_smoking"),
         d.get("occupation"), d.get("dust_exposure", False), d.get("chemical_exposure", False),
         d.get("has_asthma", False), d.get("has_tuberculosis", False),
         d.get("family_history_copd", False), d.get("current_medications"),
         d.get("alcohol_consumption"), d.get("exercise_habits")),
    )
    conn.commit()
    cur.close(); conn.close()
    return jsonify({"message": "Medical history saved"}), 201


@app.route("/api/patients/<int:patient_id>/symptoms", methods=["POST"])
@require_role("patient", "doctor")
def save_symptoms(patient_id):
    d = request.get_json(force=True)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO symptoms
           (patient_id, breathlessness, chronic_cough, wheezing, chest_tightness,
            sputum_production, fatigue, weight_loss, night_symptoms,
            walking_difficulty, sleep_disturbance, duration_of_symptoms, severity_score)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (patient_id, d.get("breathlessness", False), d.get("chronic_cough", False),
         d.get("wheezing", False), d.get("chest_tightness", False), d.get("sputum_production", False),
         d.get("fatigue", False), d.get("weight_loss", False), d.get("night_symptoms", False),
         d.get("walking_difficulty", False), d.get("sleep_disturbance", False),
         d.get("duration_of_symptoms"), d.get("severity_score", 0)),
    )
    symptom_id = cur.lastrowid
    conn.commit()
    cur.close(); conn.close()
    return jsonify({"message": "Symptoms saved", "symptom_id": symptom_id}), 201


@app.route("/api/patients/<int:patient_id>/clinical-parameters", methods=["POST"])
@require_role("patient", "doctor")
def save_clinical_parameters(patient_id):
    d = request.get_json(force=True)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO clinical_parameters
           (patient_id, spo2, heart_rate, bp_systolic, bp_diastolic,
            respiratory_rate, temperature_c, weight_kg, peak_flow, fev1_fvc_ratio)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (patient_id, d.get("spo2"), d.get("heart_rate"), d.get("bp_systolic"), d.get("bp_diastolic"),
         d.get("respiratory_rate"), d.get("temperature_c"), d.get("weight_kg"),
         d.get("peak_flow"), d.get("fev1_fvc_ratio")),
    )
    param_id = cur.lastrowid
    conn.commit()
    cur.close(); conn.close()
    return jsonify({"message": "Clinical parameters saved", "param_id": param_id}), 201


# ==================================================================
# 3. AI DIAGNOSIS ENGINE
# ==================================================================
def calculate_risk(history: dict, symptoms: dict, params: dict) -> dict:
    """
    Rule-based COPD risk scoring engine (v1).
    Swap this function's internals for a trained model (e.g. scikit-learn /
    XGBoost) without changing the API contract — inputs/outputs stay identical.
    """
    score = 0

    smoke_map = {"Never": 0, "Former": 1, "Current": 2}
    score += smoke_map.get(history.get("smoking_status", "Never"), 0) * 12
    score += min((history.get("pack_years") or 0) / 2, 20)
    score += 6 if history.get("dust_exposure") else 0
    score += 6 if history.get("chemical_exposure") else 0
    score += 5 if history.get("has_asthma") else 0
    score += 7 if history.get("family_history_copd") else 0

    symptom_flags = ["breathlessness", "chronic_cough", "wheezing", "chest_tightness",
                      "sputum_production", "fatigue", "weight_loss", "night_symptoms",
                      "walking_difficulty", "sleep_disturbance"]
    score += sum(1 for f in symptom_flags if symptoms.get(f)) * 4

    duration_map = {"<2 weeks": 0, "2 weeks-3 months": 1, ">3 months": 2}
    score += duration_map.get(symptoms.get("duration_of_symptoms"), 0) * 6
    score += (symptoms.get("severity_score") or 0) * 2

    emergency = False
    spo2 = params.get("spo2")
    if spo2 is not None:
        if spo2 < 90:
            score += 30
            emergency = True
        elif spo2 < 94:
            score += 18
        elif spo2 < 96:
            score += 8

    hr = params.get("heart_rate")
    if hr and hr > 110:
        score += 10
        if hr > 130:
            emergency = True

    rr = params.get("respiratory_rate")
    if rr and rr > 24:
        score += 8

    risk_score = max(0, min(100, round(score)))
    health_score = 100 - risk_score
    category = "Low" if risk_score < 30 else "Moderate" if risk_score < 60 else "High"

    return {
        "risk_score": risk_score,
        "health_score": health_score,
        "risk_category": category,
        "emergency": emergency,
    }


@app.route("/api/patients/<int:patient_id>/diagnose", methods=["POST"])
@require_role("patient", "doctor")
def diagnose(patient_id):
    """Runs the AI risk engine on the patient's latest history/symptoms/params
    and persists the diagnosis row + raises an emergency alert if needed."""
    conn = get_db()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM medical_history WHERE patient_id=%s ORDER BY history_id DESC LIMIT 1", (patient_id,))
    history = cur.fetchone() or {}
    cur.execute("SELECT * FROM symptoms WHERE patient_id=%s ORDER BY symptom_id DESC LIMIT 1", (patient_id,))
    symptoms = cur.fetchone() or {}
    cur.execute("SELECT * FROM clinical_parameters WHERE patient_id=%s ORDER BY param_id DESC LIMIT 1", (patient_id,))
    params = cur.fetchone() or {}

    result = calculate_risk(history, symptoms, params)

    cur.execute(
        """INSERT INTO diagnosis (patient_id, symptom_id, param_id, risk_score,
           health_score, risk_category, ai_notes) VALUES (%s,%s,%s,%s,%s,%s,%s)""",
        (patient_id, symptoms.get("symptom_id"), params.get("param_id"),
         result["risk_score"], result["health_score"], result["risk_category"],
         f"Auto-generated by rule-engine-v1"),
    )
    diagnosis_id = cur.lastrowid

    if result["emergency"]:
        cur.execute(
            """INSERT INTO emergency_alerts (patient_id, trigger_reason, spo2, heart_rate, ai_risk_score)
               VALUES (%s,%s,%s,%s,%s)""",
            (patient_id, "Critical vitals detected during AI diagnosis",
             params.get("spo2"), params.get("heart_rate"), result["risk_score"]),
        )

    conn.commit()
    cur.close(); conn.close()

    result["diagnosis_id"] = diagnosis_id
    return jsonify(result), 201


# ==================================================================
# 4. DAILY MONITORING
# ==================================================================
@app.route("/api/patients/<int:patient_id>/monitoring", methods=["POST"])
@require_role("patient")
def log_monitoring(patient_id):
    d = request.get_json(force=True)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO daily_monitoring
           (patient_id, log_date, spo2, heart_rate, bp_systolic, bp_diastolic,
            weight_kg, steps_walked, sleep_hours, medication_taken, mood,
            exercise_minutes, water_intake_ml)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON DUPLICATE KEY UPDATE spo2=VALUES(spo2), heart_rate=VALUES(heart_rate),
             steps_walked=VALUES(steps_walked), sleep_hours=VALUES(sleep_hours)""",
        (patient_id, d.get("log_date"), d.get("spo2"), d.get("heart_rate"),
         d.get("bp_systolic"), d.get("bp_diastolic"), d.get("weight_kg"),
         d.get("steps_walked"), d.get("sleep_hours"), d.get("medication_taken", False),
         d.get("mood"), d.get("exercise_minutes", 0), d.get("water_intake_ml", 0)),
    )
    conn.commit()
    cur.close(); conn.close()
    return jsonify({"message": "Monitoring entry saved"}), 201


@app.route("/api/patients/<int:patient_id>/monitoring", methods=["GET"])
@require_role("patient", "doctor")
def get_monitoring(patient_id):
    days = request.args.get("days", 30, type=int)
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM daily_monitoring WHERE patient_id=%s ORDER BY log_date DESC LIMIT %s",
        (patient_id, days),
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return jsonify(rows), 200


# ==================================================================
# 5. DOCTOR DASHBOARD
# ==================================================================
@app.route("/api/doctor/<int:doctor_id>/patients", methods=["GET"])
@require_role("doctor", "admin")
def doctor_patients(doctor_id):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT p.patient_id, p.full_name, p.age,
                  d.risk_category, d.risk_score, d.created_at AS last_diagnosis
           FROM patients p
           LEFT JOIN diagnosis d ON d.diagnosis_id = (
               SELECT diagnosis_id FROM diagnosis WHERE patient_id = p.patient_id
               ORDER BY created_at DESC LIMIT 1)
           WHERE p.assigned_doctor_id = %s
           ORDER BY d.risk_score DESC""",
        (doctor_id,),
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return jsonify(rows), 200


# ==================================================================
# 6. EMERGENCY ALERTS
# ==================================================================
@app.route("/api/patients/<int:patient_id>/alerts", methods=["GET"])
@require_role("patient", "doctor")
def get_alerts(patient_id):
    conn = get_db()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT * FROM emergency_alerts WHERE patient_id=%s ORDER BY created_at DESC",
        (patient_id,),
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return jsonify(rows), 200


# ==================================================================
# 7. AI CHATBOT (simple intent-matching; swap for LLM call if desired)
# ==================================================================
CHATBOT_KB = [
    (["what is copd"], "COPD is a progressive lung disease that narrows the airways, causing breathlessness, cough and mucus production over time."),
    (["symptom"], "Common symptoms: breathlessness, chronic cough, wheezing, chest tightness and sputum production."),
    (["inhaler"], "Exhale fully, seal lips on the mouthpiece, inhale slowly while pressing the canister, hold breath 10 seconds, then breathe out slowly."),
    (["emergency"], "Seek urgent care if SpO2 drops below 90%, breathlessness is severe, lips/fingertips turn blue, or there is confusion."),
]

@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    message = request.get_json(force=True).get("message", "").lower()
    for keywords, answer in CHATBOT_KB:
        if any(k in message for k in keywords):
            return jsonify({"reply": answer}), 200
    return jsonify({"reply": "I can help with COPD basics, symptoms, medication, inhaler use and emergency signs — try rephrasing your question."}), 200


# ==================================================================
# HEALTH CHECK
# ==================================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "AI LungCare API"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
