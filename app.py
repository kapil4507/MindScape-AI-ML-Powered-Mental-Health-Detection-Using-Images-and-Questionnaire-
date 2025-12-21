import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("--- Loading Models ---")

try:
    emotion_model = tf.keras.models.load_model('models/emotion_model.h5')
    EMOTION_LABELS = ['Angry', 'Happy', 'Sad', 'Surprise']
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print(" Emotion Model & Face Detector Loaded")
except Exception as e:
    print(f" Emotion Model Error: {e}")
    emotion_model = None
    face_cascade = None


try:
    hidden_model = tf.keras.models.load_model('models/face_hidden_model.h5', compile=False)
    print(" Face Hidden Model Loaded")
except Exception as e:
    print(f" Face Hidden Model Error: {e}")
    hidden_model = None

try:
    posture_model = tf.keras.models.load_model('models/bodysense_custom_model.keras')
    if os.path.exists('class_names.txt'):
        with open('class_names.txt', 'r') as f:
            POSTURE_LABELS = f.read().splitlines()
        print(" Posture Model Loaded")
    else:
        print(" Posture Model loaded, but 'class_names.txt' is missing.")
        POSTURE_LABELS = []
except Exception as e:
    print(f" Posture Model Error: {e}")
    posture_model = None


# PREPROCESSING FUNCTIONS

def get_emotion_score(img_path):
    if not emotion_model:
        return 0, "Unknown"
    
    img = cv2.imread(img_path)
    if img is None:
        return 0, "Error"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        # center crop fallback
        h, w = gray.shape
        min_dim = min(h, w)
        start_x, start_y = (w - min_dim) // 2, (h - min_dim) // 2
        face_input = gray[start_y:start_y+min_dim, start_x:start_x+min_dim]
    else:
        x, y, w, h = faces[0]
        center_x, center_y = x + w // 2, y + h // 2
        max_dim = int(max(w, h) * 1.2)
        new_x = max(0, center_x - max_dim // 2)
        new_y = max(0, center_y - max_dim // 2)
        
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        face_input = gray[new_y:new_y+max_dim, new_x:new_x+max_dim]
        
        if face_input.size == 0:
            face_input = gray
             
    try:
        resized = cv2.resize(face_input, (48, 48))
        normalized = resized.astype("float32") / 255.0
      
        input_data = normalized.reshape(1, 48, 48, 1)

        preds = emotion_model.predict(input_data, verbose=0)[0]

        print("\n=== EMOTION DEBUG ===")
        print(f"Image: {os.path.basename(img_path)}")
        print(f"Emotion labels: {EMOTION_LABELS}")
        print(f"Raw predictions: {preds}")
        print(f"Prediction values: {[(label, f'{prob:.4f}') for label, prob in zip(EMOTION_LABELS, preds)]}")
        
    
        label_idx = int(np.argmax(preds))
        label = EMOTION_LABELS[label_idx]
        max_confidence = float(preds[label_idx])
        
        print(f"Predicted emotion: {label} (confidence: {max_confidence:.4f})")

        emotion_map = {}
        for i, emotion_label in enumerate(EMOTION_LABELS):
            emotion_map[emotion_label.lower()] = float(preds[i])
        
        print(f"Emotion map: {emotion_map}")

        happy_prob = emotion_map.get('happy', 0.0)
        sad_prob = emotion_map.get('sad', 0.0)
        angry_prob = emotion_map.get('angry', 0.0)
        surprise_prob = emotion_map.get('surprise', 0.0)

        print(f"Happy: {happy_prob:.4f}, Sad: {sad_prob:.4f}, Angry: {angry_prob:.4f}, Surprise: {surprise_prob:.4f}")

        negative_strength = (sad_prob + angry_prob) / 2.0

        emotion_risk = negative_strength * 100.0
        emotion_risk = max(0.0, min(100.0, emotion_risk))  # clamp

        print(f"Negative strength: {negative_strength:.4f}")
        print(f"Final emotion risk: {emotion_risk:.2f}")
        print("===================\n")

        return emotion_risk, label

    except Exception as e:
        print(f"Emotion Error: {e}")
        import traceback
        traceback.print_exc()
        return 0, "Error"

def get_hidden_score(img_path):
    if not hidden_model: return 0, "N/A"
    
    try:
        img = cv2.imread(img_path)
        if img is None: return 0, "Error"
        resized = cv2.resize(img, (96, 96))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb.astype('float32') / 255.0, axis=0)
        
        preds = hidden_model.predict(input_data, verbose=0)[0]
        occlusions = preds[58:] 
        hidden_count = np.sum(occlusions > 0.5)
        percentage = (hidden_count / 29) * 100
        
        return percentage, f"{percentage:.1f}%"
    except Exception as e:
        print(f"Hidden Model Error: {e}")
        return 0, "Error"

def get_posture_score(img_path):
    if not posture_model: 
        print("Debug: Posture model is None")
        return 50, "Unknown"
    
    TRAIT_SENTIMENT = {
        'EyesDirect': 'Pos', 'HeadStraight': 'Pos', 'HeadTilt': 'Pos',
        'ShouldersOpen': 'Pos', 'ShouldersRelaxed': 'Pos', 'HandsOpen': 'Pos', 
        'HandsRelaxed': 'Pos', 'TorsoFrontal': 'Pos', 'Smile': 'Pos',
        
        'EyesAvoiding': 'Neg', 'EyesDowncast': 'Neg', 'HeadStiff': 'Neg',
        'HeadDropped': 'Neg', 'HeadDown': 'Neg', 'ShouldersRaised': 'Neg',
        'ShouldersSlumped': 'Neg', 'HandsHidden': 'Neg', 'HandsClenched': 'Neg',
        'ArmsCrossed': 'Neg', 'TorsoTurnedAway': 'Neg'
    }

    try:
        img = cv2.imread(img_path)
        if img is None: return 50, "Error"
        
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        input_data = np.expand_dims(resized, axis=0)

        preds = posture_model.predict(input_data, verbose=0)[0]
        
        pos_score = 0.0
        neg_score = 0.0
        found_any_match = False

        print(f"\n--- 🔍 DEBUG: Analyzing {os.path.basename(img_path)} ---")

        for i, score in enumerate(preds):
            if score > 0.10: 
                raw_label = POSTURE_LABELS[i] 
                
                clean_label = raw_label.replace('_', '').replace(' ', '').lower()
                
                sentiment = 'Neutral'
                

                for trait_key, trait_type in TRAIT_SENTIMENT.items():
                    clean_key = trait_key.replace('_', '').lower()
                    
                    if clean_key in clean_label:
                        sentiment = trait_type
                        break
                
                print(f"   Detected: {raw_label} ({score:.2f}) -> {sentiment}")

                if sentiment == 'Pos': 
                    pos_score += score
                    found_any_match = True
                elif sentiment == 'Neg': 
                    neg_score += score
                    found_any_match = True
                
        total = pos_score + neg_score
        
        if total == 0 or not found_any_match:
            print("    Result: No valid traits found. Defaulting to 50%.")
            return 50, "Neutral"

        risk_score = (neg_score / total) * 100
        
        print(f"   ✅ Result: Risk Score {risk_score:.1f}%")
        
        status = "Open" if risk_score < 40 else "Closed"
        return risk_score, status

    except Exception as e:
        print(f" Posture Error: {e}")
        return 50, "Error"

def fusion_engine(emotion_risk, hidden_risk, posture_risk, questionnaire_raw_score):
    q_risk = (questionnaire_raw_score / 48) * 100
    q_risk = min(q_risk, 100)

    final_score = (emotion_risk * 0.20) + \
                  (posture_risk * 0.20) + \
                  (hidden_risk * 0.20) + \
                  (q_risk * 0.40)

    if final_score < 30:
        persona = "The Open Book"
        desc = "Your signals reflect balance and resilience across multiple moments."
        color = "green"
    elif final_score < 60:
        persona = "The Silent Carrier"
        desc = "We detect mixed signals. Some moments show clarity, others show tension."
        color = "orange"
    else:
        persona = "The Shadowed Self"
        desc = "Your signals consistently suggest distress or withdrawal."
        color = "red"

    return {
        "score": int(final_score),
        "persona": persona,
        "description": desc,
        "color": color,
        "details": {
            "Emotion Risk": int(emotion_risk),
            "Posture Risk": int(posture_risk),
            "Hidden Risk": int(hidden_risk),
            "Self-Report Risk": int(q_risk)
        }
    }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'files[]' not in request.files:
        return "No files uploaded"
    
    files = request.files.getlist('files[]')
    
    # 🔒 EXACTLY 4 images enforcement
    if len(files) != 4:
        return "Please upload exactly 4 images for analysis."

    # 🔒 Enforce questionnaire completion
    try:
        gad_score = int(request.form.get('gad_score', -1))
        phq_score = int(request.form.get('phq_score', -1))
    except ValueError:
        return "Invalid questionnaire data"

    if gad_score < 0 or phq_score < 0:
        return "Please complete the questionnaire before uploading images."

    total_emo_risk = 0
    total_hid_risk = 0
    total_pos_risk = 0
    emotions_found = []

    print(f"--- Processing {len(files)} Images ---")

    for i, file in enumerate(files):
        if file.filename == '':
            continue
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{i}.jpg')
        file.save(filepath)
        
        e_score, e_label = get_emotion_score(filepath)
        h_score, h_label = get_hidden_score(filepath)
        p_score, p_label = get_posture_score(filepath)
        
        total_emo_risk += e_score
        total_hid_risk += h_score
        total_pos_risk += p_score
        
        if e_label != "Error":
            emotions_found.append(e_label)

    if not emotions_found:
        avg_emo_risk = 0
        avg_hid_risk = 0
        avg_pos_risk = 0
        dominant_emotion = "Analysis Failed"
    else:
        count = len(files)
        avg_emo_risk = total_emo_risk / count
        avg_hid_risk = total_hid_risk / count
        avg_pos_risk = total_pos_risk / count
        dominant_emotion = Counter(emotions_found).most_common(1)[0][0]

    total_q_score = gad_score + phq_score

    result = fusion_engine(avg_emo_risk, avg_hid_risk, avg_pos_risk, total_q_score)
    
    return render_template(
        'result.html', 
        result=result, 
        raw_data={
            "Emotion": dominant_emotion,
            "Posture": f"{int(avg_pos_risk)}% Risk",
            "Visibility": f"{int(avg_hid_risk)}% Hidden"
        }
    )


if __name__ == '__main__':
    app.run(debug=False)
