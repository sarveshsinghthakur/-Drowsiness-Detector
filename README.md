# Emotion Streamlit App

This folder contains a browser-based emotion detection app:

- Script: `emotion_streamlit.py`
- Webcam stream: `streamlit-webrtc`
- Emotion model: OpenCV DNN ONNX model (downloaded automatically on first run)

## Features

- Real-time webcam feed in browser
- Face detection with OpenCV Haar Cascade
- Emotion prediction from the largest detected face
- On-screen confidence bars for:
  - angry
  - disgust
  - fearful
  - happy
  - neutral
  - sad
  - surprised

## Requirements

- Windows (works on other OS with same Python deps)
- Python 3.12 recommended
- Webcam access in browser

Install dependencies with the same interpreter you use to run Streamlit:

```powershell
py -3.12 -m pip install --user streamlit streamlit-webrtc av opencv-python numpy
```

## Run

From `python projects` directory:

```powershell
py -3.12 -m streamlit run "Drowsy detector\emotion_streamlit.py"
```

Then open the local URL shown in terminal (usually `http://localhost:8501`), allow camera permission, and click **Start**.

## Notes

- The file `facial_expression_recognition_mobilefacenet_2022july.onnx` is cached in this folder after first run.
- Use one Python version consistently for install + run. Mixing Python 3.12 and 3.13 can cause module import errors.

## Troubleshooting

- `ModuleNotFoundError`:
  - Install package with the same interpreter used to run app, e.g.:
  ```powershell
  py -3.12 -m pip install --user streamlit-webrtc av opencv-python
  ```

- Browser shows black/empty video:
  - Check camera permissions for the site.
  - Close other apps using webcam.
  - Restart Streamlit app.

- App is slow:
  - Increase **Analyze every N frames** from sidebar.

