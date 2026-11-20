import streamlit as st
from ultralytics import YOLOWorld
from PIL import Image
import numpy as np
from collections import Counter

st.set_page_config(page_title="Kho hàng thông minh - YOLO-World", layout="centered")
st.title("📦 Hệ thống Nhận diện Hàng hóa Linh hoạt")

@st.cache_resource
def load_model():
    return YOLOWorld("yolov8s-worldv2.pt")

model = load_model()

st.sidebar.header("⚙️ Cấu hình nhận diện")
custom_classes_input = st.sidebar.text_input(
    "Nhập tên vật thể muốn tìm (tiếng Anh, cách nhau bằng dấu phẩy):",
    value="bottle, box, phone, pen, snack, cup, glasses"
)
conf_threshold = st.sidebar.slider("Ngưỡng độ tự tin (Confidence)", 0.1, 1.0, 0.35, 0.05)

class_list = [c.strip() for c in custom_classes_input.split(",") if c.strip()]
if class_list:
    model.set_classes(class_list)

input_option = st.radio("Chọn nguồn ảnh:", ("Chụp từ Camera", "Tải ảnh lên"), horizontal=True)
img_file = st.camera_input("Chụp ảnh hàng hóa") if input_option == "Chụp từ Camera" else st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if img_file is not None and class_list:
    image = Image.open(img_file).convert("RGB")
    image.thumbnail((800, 800))
    img_array = np.array(image)

    with st.spinner("Đang phân tích hình ảnh..."):
        results = model.predict(img_array, conf=conf_threshold)

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
        st.info("Không tìm thấy vật thể nào khớp với danh sách hiện tại.")
