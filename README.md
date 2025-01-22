# Speech Emotion Recognition Using RAVDESS Dataset

This project demonstrates a machine learning approach to recognizing emotions from speech using the RAVDESS dataset. The system extracts features from audio files and trains a Multi-Layer Perceptron (MLP) classifier to predict emotions.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Saving and Loading the Model](#saving-and-loading-the-model)

## Introduction
This project aims to classify speech emotions using the RAVDESS dataset. It involves preprocessing audio data, extracting relevant features (MFCC, Chroma, and Mel), training a machine learning model, and evaluating its performance.

## Dataset
The RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset is used in this project. It contains audio files with various emotional expressions. The emotions observed in this project are:
- Calm
- Happy
- Fearful
- Disgust

## Dependencies
The following libraries are required:
```bash
pip install librosa soundfile numpy sklearn pyaudio
pip install resampy
```

## File Structure
- **Dataset**: Place the RAVDESS dataset in the directory `/content/drive/MyDrive/Colab Notebooks/RAVDESS_Emotional_speech_audio`.
- **Code**: Includes feature extraction, training, and evaluation scripts.
- **Trained Model**: Saved as `modelForPrediction1.sav` for reuse.

## Feature Extraction
The `extract_feature` function extracts the following features from audio files:
1. MFCC (Mel Frequency Cepstral Coefficients)
2. Chroma
3. Mel Spectrogram

### Code Example:
```python
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result
```

## Model Training
The data is split into training and testing sets (75%-25%). A Multi-Layer Perceptron Classifier (MLP) is trained on the extracted features.

### Code Example:
```python
model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, 
                      hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train, y_train)
```

## Evaluation
The model is evaluated using accuracy and F1 scores.

### Code Example:
```python
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
```

## Usage
### Loading the Model:
```python
with open('modelForPrediction1.sav', 'rb') as f:
    loaded_model = pickle.load(f)
```

### Making Predictions:
```python
feature = extract_feature("path/to/audio.wav", mfcc=True, chroma=True, mel=True)
feature = feature.reshape(1, -1)
prediction = loaded_model.predict(feature)
print(prediction)
```

## Results
The classifier achieved an accuracy of over 90% (example value; replace with actual results from evaluation). The confusion matrix and F1 scores highlight the performance across different emotions.

## Saving and Loading the Model
The trained model is saved using Python's `pickle` library and can be reloaded for predictions without retraining.

### Saving the Model:
```python
with open('modelForPrediction1.sav', 'wb') as f:
    pickle.dump(model, f)
```

### Loading the Model:
```python
loaded_model = pickle.load(open('modelForPrediction1.sav', 'rb'))
```

---
For further enhancements or issues, feel free to contribute to the repository!

