from flask import Flask, Response, render_template
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from twilio.rest import Client

# Initialize Flask app
app = Flask(__name__, template_folder='.')

# Load the trained violence detection model
model = tf.keras.models.load_model("model.h5")  # Ensure model.h5 is in the same directory

# Twilio credentials
account_sid = 'YOUR_SID'
auth_token = 'YOUR_TOKEN'
twilio_phone_number = 'TWILIO_NO'
to_phone_number = 'YOUR_PH_NO'

def send_sms_alert():
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body='Violence has been detected by the surveillance system.',
            from_=twilio_phone_number,
            to=to_phone_number
        )
        print(f'SMS sent with SID: {message.sid}')
    except Exception as e:
        print(f'Error sending SMS: {e}')

# Open the camera
camera = cv2.VideoCapture(0)

# Frame buffer to store 16 consecutive frames (rolling window)
frame_buffer = deque(maxlen=16)

# Stability settings
last_predictions = deque(maxlen=10)
violence_threshold = 0.8  # Only detect violence if probability > 80%
min_violence_frames = 5  # Violence must be detected in at least 5 of the last 10 frames
sms_sent = False  # To prevent sending multiple SMS alerts

# Function to preprocess the frame before feeding it to the model
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize to match model input
    frame = frame / 255.0  # Normalize
    return frame

# Function to generate frames for the video feed
def generate_frames():
    global frame_buffer, last_predictions, sms_sent

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame = preprocess_frame(frame)
            frame_buffer.append(processed_frame)

            if len(frame_buffer) == 16:
                input_sequence = np.array(frame_buffer).reshape(1, 16, 64, 64, 3)
                prediction = model.predict(input_sequence)[0][0]
                last_predictions.append(prediction)

                violence_count = sum(1 for p in last_predictions if p > violence_threshold)
                if violence_count >= min_violence_frames:
                    label = "Violence Detected"
                    color = (0, 0, 255)
                    if not sms_sent:
                        send_sms_alert()
                        sms_sent = True  # Prevent multiple alerts
                else:
                    label = "No Violence"
                    color = (0, 255, 0)
                    sms_sent = False  # Reset SMS trigger if no violence detected
            else:
                label = "Analyzing..."
                color = (255, 255, 0)

            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('templates/index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
