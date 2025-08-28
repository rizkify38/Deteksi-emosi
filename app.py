import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==============================
# Load Model
# ==============================
model = YOLO("model/best.pt")

# Mapping class → label Indo
classes = {
    "anger": "Marah",
    "contempt": "Menghina",
    "disgust": "Jijik",
    "fear": "Takut",
    "happiness": "Bahagia",
    "neutrality": "Netral",
    "sadness": "Sedih",
    "surprise": "Terkejut"
}

# ==============================
# Sidebar Menu
# ==============================
st.sidebar.title("📌 Menu")
menu = st.sidebar.radio("Pilih Mode", ["Deteksi Gambar", "Deteksi Kamera"])

# ==============================
# Mode 1: Deteksi dari Gambar
# ==============================
if menu == "Deteksi Gambar":
    st.title("📷 Deteksi Emosi dari Gambar")
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        st.image(img, caption="Gambar yang diunggah", use_column_width=True)

        if st.button("Deteksi Emosi"):
            results = model.predict(img_np, verbose=False)
            if results and len(results[0].probs) > 0:
                cls_id = int(results[0].probs.top1)
                conf = float(results[0].probs.top1conf)
                class_name = list(classes.keys())[cls_id]
                label_indo = classes[class_name]

                st.success(f"✅ Emosi Terdeteksi: **{label_indo}** ({conf:.2f})")

            # Tampilkan gambar dengan anotasi
            annotated_img = results[0].plot()
            st.image(annotated_img, caption="Hasil Deteksi", use_column_width=True)

# ==============================
# Mode 2: Deteksi dari Kamera
# ==============================
elif menu == "Deteksi Kamera":
    st.title("🎥 Deteksi Emosi Real-Time (Kamera)")

    class EmotionTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(img, verbose=False)
            if results and len(results[0].probs) > 0:
                cls_id = int(results[0].probs.top1)
                conf = float(results[0].probs.top1conf)
                class_name = list(classes.keys())[cls_id]
                label_indo = classes[class_name]

                cv2.putText(img, f"{label_indo} {conf:.2f}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)
            return img

    webrtc_streamer(key="emotion-detection",
                    video_transformer_factory=EmotionTransformer,
                    media_stream_constraints={"video": True, "audio": False})
