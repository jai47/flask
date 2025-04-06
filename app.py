from flask import Flask, render_template, request, jsonify
import pickle
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from sklearn.decomposition import PCA
import torch
import librosa
import numpy as np
from flask_cors import CORS

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Set to evaluation mode to avoid training
model.eval()

# Set a smaller sample rate to reduce memory usage
TARGET_SAMPLE_RATE = 16000
CHUNK_DURATION = 30  # Chunk duration in seconds
# Load PCA model
pca = PCA(n_components=2, svd_solver='full')



model_path = 'model.pkl'
with open(model_path, 'rb') as f:
    clf_model = pickle.load(f)


app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    def extract_features_from_wav2vec2(audio_input):
        inputs = processor(audio_input, return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.squeeze().cpu().numpy()

    def process_in_chunks(file_path, chunk_duration=CHUNK_DURATION, sr=TARGET_SAMPLE_RATE):
        try:
            # Load the file with the target sample rate
            y, sr = librosa.load(file_path, sr=sr)
            
            # Get the total duration of the audio file
            total_duration = librosa.get_duration(y=y, sr=sr)
            
            features_list = []  # Store features of all chunks
            
            # Process the file in chunks
            for start in range(0, 414, chunk_duration):
                end = min(start + chunk_duration, 414)
                y_chunk = y[int(start * sr):int(end * sr)]
                
                # Extract features for this chunk using Wav2Vec2
                chunk_features = extract_features_from_wav2vec2(y_chunk)
                chunk_features = chunk_features.T  # Transpose for PCA
                
                # Apply PCA and flatten the features
                pca_features = pca.fit_transform(chunk_features)
                pca_features = pca_features.flatten().reshape(1, -1)  # Ensure 2D shape
                
                features_list.append(pca_features)

            if features_list:
                # Stack features along the first axis
                return np.hstack(features_list)
            else:
                return None
        
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            return None


    print(f"Extracted features shape: {file.name}")
    features = process_in_chunks(file)
    # Optionally save the file or process it directly
    # file.save("uploaded_audio.wav")

    # Make prediction
    prediction = clf_model.predict(features)

    # Get the class with the highest probability
    probabilities = clf_model.predict_proba(features)
    # Return the prediction result
    print(f"Prediction: {prediction}")
    print(f"Probabilities: {probabilities}")

    return jsonify({"status": "success", "filename": file.filename, "accuracy": str(probabilities[0][0]), "prediction": str(prediction[0])})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
# Note: Make sure to install the required packages