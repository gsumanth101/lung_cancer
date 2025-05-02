from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'your_secret_key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database connection string
DATABASE_URL = "#"

# Database connection
def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

# Create tables if they don't exist
def create_tables():
    create_users_table = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(255) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    );
    """
    create_submissions_table = """
    CREATE TABLE IF NOT EXISTS submissions (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        image_path TEXT NOT NULL,
        prediction VARCHAR(50) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(create_users_table)
        cur.execute(create_submissions_table)
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error creating tables: {str(e)}")

# Call the function to create tables at startup
create_tables()

# Load the model
model = tf.keras.models.load_model('lung_cancer_detection_model.h5')

def preprocess_image(image_path):
    target_size = model.input_shape[1:3]
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1, *target_size, 3))
    return img_array

@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                img_array = preprocess_image(filepath)
                predictions = model.predict(img_array)
                classes = ['Normal', 'Benign', 'Malignant']
                predicted_class_index = np.argmax(predictions)
                predicted_class = classes[predicted_class_index]

                # Store submission in the database
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO submissions (user_id, image_path, prediction) VALUES (%s, %s, %s)",
                    (session['user_id'], filepath, predicted_class)
                )
                conn.commit()
                cur.close()
                conn.close()

                flash(f'Prediction: {predicted_class}', 'success')
                return redirect(url_for('dashboard'))
            except Exception as e:
                flash(f"Error during prediction: {str(e)}", 'danger')
                return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('home'))

    if 'image' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            classes = ['Normal', 'Benign', 'Malignant']
            predicted_class_index = np.argmax(predictions)
            predicted_class = classes[predicted_class_index]

            # Store submission in the database
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO submissions (user_id, image_path, prediction) VALUES (%s, %s, %s)",
                (session['user_id'], filepath, predicted_class)
            )
            conn.commit()
            cur.close()
            conn.close()

            # Render the result.html template with prediction and image URL
            return render_template('result.html', prediction=predicted_class, image_url=filepath)
        except Exception as e:
            flash(f"Error during prediction: {str(e)}", 'danger')
            return redirect(url_for('dashboard'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            cur.close()
            conn.close()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('home'))
        except Exception as e:
            flash(f"Error: {str(e)}", 'danger')
            return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    except Exception as e:
        flash(f"Error: {str(e)}", 'danger')
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('home'))

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM submissions WHERE user_id = %s", (session['user_id'],))
        submissions = cur.fetchall()
        cur.close()
        conn.close()
        return render_template('dashboard.html', submissions=submissions)
    except Exception as e:
        flash(f"Error: {str(e)}", 'danger')
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)