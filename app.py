import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from collections import Counter

st.set_page_config(page_title="Kho hàng thông minh - Nhóm 5", layout="centered")
st.title("📦 Hệ thống Nhận diện Hàng hóa Thông minh")

# Chỉ giữ đúng 1 model trong RAM tại một thời điểm để chống tràn 1GB RAM
@st.cache_resource(max_entries=1)
def load_model(model_path):
    return YOLO(model_path)

st.sidebar.header("⚙️ Cấu hình hệ thống")
mode = st.sidebar.radio(
    "Chọn chế độ nhận diện:",
    ("Kho hàng chuyên dụng (9 món VN)", "Vật thể phổ thông (80 loại)")
)

model_path = "best.pt" if mode == "Kho hàng chuyên dụng (9 món VN)" else "yolov8n.pt"
default_conf = 0.65 if model_path == "best.pt" else 0.40
conf_threshold = st.sidebar.slider("Ngưỡng độ tự tin (Confidence)", 0.1, 1.0, default_conf, 0.05)

model = load_model(model_path)

input_option = st.radio("Chọn nguồn ảnh:", ("Chụp từ Camera", "Tải ảnh lên"), horizontal=True)
img_file = st.camera_input("Chụp ảnh hàng hóa") if input_option == "Chụp từ Camera" else st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    image.thumbnail((800, 800))
    with st.spinner("Đang phân tích hình ảnh..."):
    results = model.predict(image, conf=conf_threshold)

    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Kết quả nhận diện", channels="BGR", use_container_width=True)

    boxes = results[0].boxes
    if len(boxes) > 0:
        detected_names = [model.names[int(cls)] for cls in boxes.cls]
        counts = Counter(detected_names)
        st.subheader("📊 Thống kê kiểm kê:")
        for name, count in counts.items():
            st.write(f"- **{name}**: {count} cái")
    else:
        st.info("Không phát hiện vật thể nào đạt ngưỡng tự tin.")
