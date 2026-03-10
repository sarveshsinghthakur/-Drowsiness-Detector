from pathlib import Path
from urllib.request import urlretrieve

import av
import cv2
import numpy as np
import streamlit as st

try:
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
except ImportError:
    st.error("Missing dependency: streamlit-webrtc")
    st.code("py -3.12 -m pip install streamlit-webrtc av")
    st.stop()

MODEL_URL = (
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/"
    "facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx"
)
MODEL_PATH = Path(__file__).with_name("facial_expression_recognition_mobilefacenet_2022july.onnx")

DEFAULT_EMOTIONS = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


@st.cache_resource
def ensure_model_path() -> str:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self, model_path: str, analyze_every_n_frames: int):
        self.analyze_every_n_frames = max(1, analyze_every_n_frames)
        self.frame_count = 0
        self.last_emotion = "neutral"
        self.last_confidence = 0.0
        self.last_probs = {emotion: 0.0 for emotion in DEFAULT_EMOTIONS}

        self.model = cv2.dnn.readNet(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / np.sum(exp)

    @staticmethod
    def _draw_meter(frame, x, y, w, h, pct, label, fg):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 35, 35), -1)
        fill = int(w * max(0.0, min(1.0, pct)))
        if fill > 0:
            cv2.rectangle(frame, (x, y), (x + fill, y + h), fg, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)
        cv2.putText(frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

    def _predict(self, face_bgr: np.ndarray):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (112, 112))
        x = rgb.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        blob = cv2.dnn.blobFromImage(x)
        self.model.setInput(blob)
        logits = self.model.forward().reshape(-1)
        probs = self._softmax(logits) * 100.0
        idx = int(np.argmax(probs))
        return DEFAULT_EMOTIONS[idx], float(probs[idx]), {
            emotion: float(probs[i]) for i, emotion in enumerate(DEFAULT_EMOTIONS)
        }

    def _draw_overlay(self, frame):
        _, w = frame.shape[:2]
        panel_w = 340
        panel_h = 290
        x0 = max(8, w - panel_w - 8)
        y0 = 8

        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.putText(frame, "EMOTION", (x0 + 12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
        cv2.putText(
            frame,
            f"{self.last_emotion.upper()} ({self.last_confidence:.1f}%)",
            (x0 + 12, y0 + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 220),
            2,
        )

        colors = {
            "happy": (0, 180, 0),
            "surprised": (0, 180, 255),
            "neutral": (0, 170, 170),
        }

        y = y0 + 88
        bar_h = 18
        gap = 10
        for emotion in DEFAULT_EMOTIONS:
            prob = self.last_probs.get(emotion, 0.0)
            fg = colors.get(emotion, (0, 120, 255))
            self._draw_meter(
                frame,
                x0 + 12,
                y,
                panel_w - 24,
                bar_h,
                prob / 100.0,
                f"{emotion:8s} {prob:5.1f}%",
                fg,
            )
            y += bar_h + gap

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
        )

        if len(faces) > 0 and self.frame_count % self.analyze_every_n_frames == 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = image[y : y + h, x : x + w]
            try:
                emotion, confidence, probs = self._predict(face_crop)
                self.last_emotion = emotion
                self.last_confidence = confidence
                self.last_probs = probs
            except Exception:
                pass

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image,
                self.last_emotion,
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        self._draw_overlay(image)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def main():
    st.set_page_config(page_title="Emotion Detector", layout="wide")
    st.title("Emotion Detection in Browser")
    st.write("Start stream and allow camera access in your browser.")

    analyze_every = st.sidebar.slider("Analyze every N frames", min_value=1, max_value=15, value=4, step=1)

    with st.spinner("Preparing emotion model..."):
        model_path = ensure_model_path()

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="emotion-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: EmotionVideoProcessor(model_path, analyze_every),
        async_processing=True,
    )

    st.caption("Model file is downloaded once and cached in this folder.")


if __name__ == "__main__":
    main()
