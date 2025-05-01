import cv2
import numpy as np
import tensorflow as tf
import json
import openai
import os
from pyngrok import ngrok
from flask import Flask, request, jsonify, render_template, redirect, url_for, session

# Labels
class_names = ["CFP", "OCT"]

cfp_class_names = ['Normal', 'cataract', 'diabetic_retinopathy', 'glaucoma']
oct_class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# إعداد Flask
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)


os.makedirs("model", exist_ok=True)


cfp_model = tf.keras.models.load_model("model/CFP_vgg19_model.keras")
oct_model = tf.keras.models.load_model("model/ttry2.h5")
oct_cfp_model = tf.keras.models.load_model("model/oct_cfp_vgg19_model (2).h5")

# preprocessing image
def refine_cropping(image):
    image = tf.cast(image, tf.float32)
    histogram = tf.histogram_fixed_width(image, [0, 255], nbins=256)
    peak_intensity = tf.argmax(histogram[-50:]) + 175
    threshold = tf.cast(peak_intensity - 10, tf.float32)
    binary_mask = tf.where(image >= threshold, 255.0, 0.0)

    non_white_coords = tf.where(binary_mask < 255.0)
    ymin = tf.cast(tf.reduce_min(non_white_coords[:, 0]), tf.int32)
    ymax = tf.cast(tf.reduce_max(non_white_coords[:, 0]), tf.int32)
    xmin = tf.cast(tf.reduce_min(non_white_coords[:, 1]), tf.int32)
    xmax = tf.cast(tf.reduce_max(non_white_coords[:, 1]), tf.int32)

    image_shape = tf.shape(image)
    margin = tf.minimum(20, tf.minimum(ymin, xmin))
    ymin = tf.maximum(0, ymin - margin)
    ymax = tf.minimum(image_shape[0], ymax + margin)
    xmin = tf.maximum(0, xmin - margin)
    xmax = tf.minimum(image_shape[1], xmax + margin)

    cropped_image = tf.image.crop_to_bounding_box(tf.cast(image, tf.uint8), ymin, xmin, ymax - ymin, xmax - xmin)
    return cropped_image

def preprocess_cfp_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_oct_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image

from tensorflow.keras.applications.vgg19 import preprocess_input

def preprocess_standard_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def predict_image_type(image_path):
    image = preprocess_standard_image(image_path)
    prediction = oct_cfp_model.predict(image)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)
    print(f"Predicted Class: {class_names[predicted_index]} with confidence: {confidence}")
    if confidence < 0.7:
        print("Warning: The model's prediction is not very confident.")
    return class_names[predicted_index]

def predict_diagnosis(image_path, model, labels, image_type):
    if image_type == "CFP":
        image = preprocess_cfp_image(image_path)
    else:
        image = preprocess_oct_image(image_path)
    if image is None:
        print("Error: Image processing failed.")
        return None, None
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return labels[predicted_class], predictions[0]

def generate_recommendations(diagnosis, user_responses):
    os.environ["OPENAI_API_KEY"] = "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"
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    You are an expert ophthalmologist AI assistant. Your task is to provide **personalized eye health recommendations** based on the user’s diagnosis and lifestyle factors.
    **Diagnosis:** {diagnosis}
    **User Details:**
    - Age: {user_responses.get("age", "Unknown")}
    - Gender: {user_responses.get("gender", "Unknown")}
    - Healthy Diet: {user_responses.get("diet", "Unknown")}
    - Screen Time: {user_responses.get("screen_time", "Unknown")}
    - Smoking: {user_responses.get("smoke", "Unknown")}
    - Dry Eyes: {user_responses.get("dry_eyes", "Unknown")}
    - Contact Lenses: {user_responses.get("contact_lenses", "Unknown")}
    - Location: {user_responses.get("location", "Unknown")}
    - Chronic Diseases: {user_responses.get("chronic_diseases", "Unknown")}

    Based on this information:
    1. **Explain the condition in simple terms.**
    2. **Medical Recommendations:** Provide three specific treatments or medications.
    3. **Lifestyle Modifications:** Suggest three personalized changes based on the user’s habits.
    4. **Precautionary Note:** Provide a final warning or care tip.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content

# HTML pages
@app.route('/')
@app.route('/sn_new')
def sn_new():
    return render_template("SN_new.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/page')
def page():
    return render_template("page_3.html")

@app.route('/scan')
def scan():
    return render_template("scan.html")

@app.route('/help')
def help():
    return "<h1>Help page not created yet</h1>"

# image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# predict point
@app.route('/predict', methods=['POST'])
def predict():
    print("DEBUG - Request files:", request.files)
    print("DEBUG - Request form:", request.form)
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400
    image_file = request.files['image']
    image_bytes = image_file.read()
    print("DEBUG - Image bytes length:", len(image_bytes))  # الحين عادي نطبع
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_path = "uploaded_image.jpg"
        image.save(image_path)
    except Exception as e:
        print(f"Image processing error: {str(e)}")  # أطبع الخطأ الحقيقي
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 400

    # Step 1: Detect image type
    image_type = predict_image_type(image_path)

    # Step 2: Predict diagnosis
    try:
        if image_type == "CFP":
            diagnosis, scores = predict_diagnosis(image_path, cfp_model, cfp_class_names, image_type)
        else:
            diagnosis, scores = predict_diagnosis(image_path, oct_model, oct_class_names, image_type)
    except Exception as e:
        return jsonify({'error': f'Diagnosis model failed: {str(e)}'}), 500

    # Step 3: Collect user responses
    user_responses = {
        'age': request.form.get('age'),
        'gender': request.form.get('gender'),
        'diet': request.form.get('diet'),
        'screen_time': request.form.get('screen_time'),
        'smoke': request.form.get('smoke'),
        'dry_eyes': request.form.get('dry_eyes'),
        'contact_lenses': request.form.get('contact_lenses'),
        'location': request.form.get('location'),
        'chronic_diseases': request.form.get('chronic_diseases')
    }

    # Step 4: Generate recommendations
    try:
        recommendations = generate_recommendations(diagnosis, user_responses)
    except Exception as e:
        recommendations = {'error': f'Recommendation model failed: {str(e)}'}

    response_data = {
        'image_type': image_type,
        'diagnosis': diagnosis,
        'scores': scores.tolist() if isinstance(scores, np.ndarray) else scores,
        'recommendations': recommendations
    }

 
    print(f"DEBUG - Response: {response_data}")

    return jsonify(response_data)

if __name__ == "__main__":
    port = 5000
    public_url = ngrok.connect(port)
    print(f" * ngrok tunnel link: {public_url}")
    app.run(port=port)
