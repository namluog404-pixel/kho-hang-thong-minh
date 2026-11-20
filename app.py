import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Kho Hàng Thông Minh - Nhóm 5", layout="centered")
st.title("Demo AI Kiểm Kê - Kho Hàng Thông Minh (Nhóm 5)")
st.write("Tải ảnh lên hoặc chụp trực tiếp để hệ thống tự động nhận diện và đếm số lượng vật thể.")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

option = st.radio("Chọn nguồn ảnh:", ("Tải ảnh lên", "Chụp bằng Camera"))
img_file = st.file_uploader("Chọn file ảnh...", type=["jpg", "jpeg", "png"]) if option == "Tải ảnh lên" else st.camera_input("Chụp ảnh hàng hóa")

if img_file is not None:
    image = Image.open(img_file)
    img_array = np.array(image)

    with st.spinner("Đang nhận diện..."):
        results = model(img_array, conf=0.5, verbose=False)
        annotated_img = results[0].plot()
        annotated_img = annotated_img[:, :, ::-1]  

    st.image(annotated_img, caption="Kết quả nhận diện", use_container_width=True)

    counts = {}
    for box in results[0].boxes:
        name = model.names[int(box.cls[0])]
        counts[name] = counts.get(name, 0) + 1

    st.subheader("Thống kê kiểm kê:")
    if not counts:
        st.info("Không phát hiện vật thể nào.")
    else:
        for k, v in counts.items():
            st.write(f"- **{k}**: {v} cái")
