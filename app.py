import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
from collections import Counter

st.set_page_config(page_title="Kho hàng thông minh - Bản Mở Rộng", layout="wide")
st.title("📦 Hệ thống Giám sát Kệ hàng Thông minh (Bản Mở Rộng 80 Vật thể)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
all_classes = sorted(list(model.names.values()))


if "prev_counts" not in st.session_state:
    st.session_state.prev_counts = Counter()


st.sidebar.header("⚙️ Cấu hình Kệ hàng (Lưới 2x2)")
st.sidebar.caption("Quy định loại vật phẩm cố định cho từng ô kệ:")

zone_rules = {
    "Ô 1 (Trên-Trái)": st.sidebar.selectbox("Ô 1 (Trên-Trái):", all_classes, index=all_classes.index("bottle")),
    "Ô 2 (Trên-Phải)": st.sidebar.selectbox("Ô 2 (Trên-Phải):", all_classes, index=all_classes.index("cup")),
    "Ô 3 (Dưới-Trái)": st.sidebar.selectbox("Ô 3 (Dưới-Trái):", all_classes, index=all_classes.index("book")),
    "Ô 4 (Dưới-Phải)": st.sidebar.selectbox("Ô 4 (Dưới-Phải):", all_classes, index=all_classes.index("cell phone")),
}

st.sidebar.divider()
conf_threshold = st.sidebar.slider("Ngưỡng độ tự tin (Confidence)", 0.1, 1.0, 0.40, 0.05)
ignore_person = st.sidebar.checkbox("Bỏ qua người (person) khi kiểm kê", value=True)

if st.sidebar.button("🔄 Đặt lại lịch sử quét"):
    st.session_state.prev_counts = Counter()
    st.sidebar.success("Đã xóa trạng thái cũ.")


input_option = st.radio("Chọn nguồn ảnh:", ("Chụp từ Camera", "Tải ảnh lên"), horizontal=True)
img_file = st.camera_input("Chụp ảnh kệ hàng") if input_option == "Chụp từ Camera" else st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    image.thumbnail((800, 800))

    with st.spinner("Đang phân tích vị trí và kiểm kê..."):
        results = model.predict(image, conf=conf_threshold)

    
    annotated_img = results[0].plot()
    h, w, _ = annotated_img.shape
    mid_x, mid_y = w // 2, h // 2

    
    cv2.line(annotated_img, (mid_x, 0), (mid_x, h), (0, 255, 255), 2)
    cv2.line(annotated_img, (0, mid_y), (w, mid_y), (0, 255, 255), 2)

    
    cv2.putText(annotated_img, f"O 1: {zone_rules['Ô 1 (Trên-Trái)']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated_img, f"O 2: {zone_rules['Ô 2 (Trên-Phải)']}", (mid_x + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated_img, f"O 3: {zone_rules['Ô 3 (Dưới-Trái)']}", (10, mid_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated_img, f"O 4: {zone_rules['Ô 4 (Dưới-Phải)']}", (mid_x + 10, mid_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    
    current_counts = Counter()
    misplaced_alerts = []

    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        if ignore_person and cls_name == "person":
            continue

        current_counts[cls_name] += 1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        
        if cx < mid_x and cy < mid_y:
            zone = "Ô 1 (Trên-Trái)"
        elif cx >= mid_x and cy < mid_y:
            zone = "Ô 2 (Trên-Phải)"
        elif cx < mid_x and cy >= mid_y:
            zone = "Ô 3 (Dưới-Trái)"
        else:
            zone = "Ô 4 (Dưới-Phải)"

        
        cv2.circle(annotated_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        
        expected = zone_rules[zone]
        if cls_name != expected:
            misplaced_alerts.append(f"⚠️ **{cls_name}** đang nằm sai tại **{zone}** (Quy định ô này: `{expected}`)")

    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(annotated_img, caption="Bản đồ giám sát kệ hàng (Chấm đỏ = Tâm vật thể)", channels="BGR", use_container_width=True)

    with col2:
        st.subheader("📊 Thống kê kiểm kê")
        if current_counts:
            for item, count in current_counts.items():
                st.write(f"- **{item}**: {count} cái")
        else:
            st.info("Chưa phát hiện hàng hóa nào.")

        st.divider()
        st.subheader("🚨 Cảnh báo vị trí (Xếp lộn xộn)")
        if misplaced_alerts:
            for alert in misplaced_alerts:
                st.error(alert)
        else:
            st.success("✅ Tất cả vật phẩm đều nằm đúng ô quy định.")

        st.divider()
        st.subheader("📉 Biến động kho (So với lần quét trước)")
        taken_items = []
        for item, prev_qty in st.session_state.prev_counts.items():
            curr_qty = current_counts.get(item, 0)
            if curr_qty < prev_qty:
                taken_items.append(f"🔻 **{item}**: đã bị lấy đi **{prev_qty - curr_qty}** cái (còn {curr_qty})")

        if taken_items:
            for msg in taken_items:
                st.warning(msg)
        else:
            st.write("Không có vật phẩm nào bị lấy đi.")

    
    st.session_state.prev_counts = current_counts
