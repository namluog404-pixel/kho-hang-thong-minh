import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter

st.set_page_config(page_title="Kho hàng thông minh - Nhóm 5", layout="centered")
st.title("📦 Hệ thống Nhận diện Vật thể Thông minh")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.sidebar.header("⚙️ Cấu hình nhận diện")
conf_threshold = st.sidebar.slider("Ngưỡng độ tự tin (Confidence)", 0.1, 1.0, 0.40, 0.05)

input_option = st.radio("Chọn nguồn ảnh:", ("Chụp từ Camera", "Tải ảnh lên"), horizontal=True)
img_file = st.camera_input("Chụp ảnh vật thể") if input_option == "Chụp từ Camera" else st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

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
