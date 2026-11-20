from collections import Counter
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import cv2
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Kho hàng thông minh - Nhóm 5", layout="wide")
st.title("📦 Hệ thống Giám sát Kệ hàng Thông minh")

# Từ điển tiếng Việt ưu tiên cho các nhãn COCO thường gặp trong kiểm kê / vật dụng
VI_LABELS = {
    "person": "người",
    "bicycle": "xe đạp",
    "car": "ô tô",
    "motorcycle": "xe máy",
    "airplane": "máy bay",
    "bus": "xe buýt",
    "train": "tàu hỏa",
    "truck": "xe tải",
    "boat": "thuyền",
    "traffic light": "đèn giao thông",
    "fire hydrant": "trụ cứu hỏa",
    "stop sign": "biển báo dừng",
    "parking meter": "đồng hồ đỗ xe",
    "bench": "ghế dài",
    "bird": "chim",
    "cat": "mèo",
    "dog": "chó",
    "horse": "ngựa",
    "sheep": "cừu",
    "cow": "bò",
    "elephant": "voi",
    "bear": "gấu",
    "zebra": "ngựa vằn",
    "giraffe": "hươu cao cổ",
    "backpack": "ba lô",
    "umbrella": "ô / dù",
    "handbag": "túi xách",
    "tie": "cà vạt",
    "suitcase": "vali",
    "frisbee": "đĩa ném",
    "skis": "ván trượt tuyết",
    "snowboard": "ván trượt tuyết đơn",
    "sports ball": "bóng thể thao",
    "kite": "diều",
    "baseball bat": "gậy bóng chày",
    "baseball glove": "găng bóng chày",
    "skateboard": "ván trượt",
    "surfboard": "ván lướt sóng",
    "tennis racket": "vợt tennis",
    "bottle": "chai",
    "wine glass": "ly thủy tinh",
    "cup": "cốc / ly",
    "fork": "nĩa",
    "knife": "dao",
    "spoon": "thìa",
    "bowl": "bát / tô",
    "banana": "chuối",
    "apple": "táo",
    "sandwich": "bánh mì kẹp",
    "orange": "cam",
    "broccoli": "bông cải xanh",
    "carrot": "cà rốt",
    "hot dog": "xúc xích kẹp",
    "pizza": "pizza",
    "donut": "bánh rán",
    "cake": "bánh ngọt",
    "chair": "ghế",
    "couch": "ghế sofa",
    "potted plant": "chậu cây",
    "bed": "giường",
    "dining table": "bàn ăn",
    "toilet": "bồn cầu",
    "tv": "tivi",
    "laptop": "laptop",
    "mouse": "chuột máy tính",
    "remote": "điều khiển",
    "keyboard": "bàn phím",
    "cell phone": "điện thoại",
    "microwave": "lò vi sóng",
    "oven": "lò nướng",
    "toaster": "máy nướng bánh mì",
    "sink": "bồn rửa",
    "refrigerator": "tủ lạnh",
    "book": "sách",
    "clock": "đồng hồ",
    "vase": "bình hoa",
    "scissors": "kéo",
    "teddy bear": "gấu bông",
    "hair drier": "máy sấy tóc",
    "toothbrush": "bàn chải đánh răng",
}


def get_vn_now() -> datetime:
    try:
        return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    except ZoneInfoNotFoundError:
        return datetime.now(timezone(timedelta(hours=7)))


def to_vi_label(cls_name: str, use_vi: bool) -> str:
    if not use_vi:
        return cls_name
    return VI_LABELS.get(cls_name, cls_name)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


def draw_label(img, text, pos):
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness)


model = load_model()
raw_classes = sorted(list(model.names.values()))

# --- THANH CẤU HÌNH ---
st.sidebar.header("⚙️ Cấu hình Kệ hàng (Lưới 2x2)")
lang_mode = st.sidebar.radio("Ngôn ngữ nhãn:", ("Tiếng Việt", "Tiếng Anh"), horizontal=True)
use_vi = lang_mode == "Tiếng Việt"

display_to_raw = {to_vi_label(c, use_vi): c for c in raw_classes}
display_classes = list(display_to_raw.keys())

default_raw = {
    "Ô 1 (Trên-Trái)": "bottle",
    "Ô 2 (Trên-Phải)": "cup",
    "Ô 3 (Dưới-Trái)": "book",
    "Ô 4 (Dưới-Phải)": "cell phone",
}

if "prev_counts" not in st.session_state:
    st.session_state.prev_counts = Counter()
if "history" not in st.session_state:
    st.session_state.history = []

zone_rules = {}
for zone_name, raw_default in default_raw.items():
    default_disp = to_vi_label(raw_default, use_vi)
    default_idx = display_classes.index(default_disp) if default_disp in display_classes else 0
    selected_disp = st.sidebar.selectbox(f"{zone_name}:", display_classes, index=default_idx)
    zone_rules[zone_name] = display_to_raw[selected_disp]

st.sidebar.divider()
conf_threshold = st.sidebar.slider("Ngưỡng độ tự tin (Confidence)", 0.1, 1.0, 0.40, 0.05)
ignore_person = st.sidebar.checkbox("Bỏ qua người (person) khi kiểm kê", value=True)

if st.sidebar.button("🔄 Đặt lại lịch sử quét"):
    st.session_state.prev_counts = Counter()
    st.session_state.history = []
    st.sidebar.success("Đã làm mới toàn bộ dữ liệu.")

# --- GIAO DIỆN CHÍNH ---
input_option = st.radio(
    "Chọn nguồn ảnh:", ("Chụp từ Camera", "Tải ảnh lên"), horizontal=True
)
img_file = (
    st.camera_input("Chụp ảnh kệ hàng")
    if input_option == "Chụp từ Camera"
    else st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])
)

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    image.thumbnail((800, 800))

    with st.spinner("Đang phân tích vị trí và kiểm kê..."):
        results = model.predict(image, conf=conf_threshold)

    # Vẽ khung bao từ YOLO rồi gắn nhãn tiếng Việt/Anh chuẩn hóa theo lựa chọn
    annotated_img = results[0].orig_img.copy()
    h, w, _ = annotated_img.shape
    mid_x, mid_y = w // 2, h // 2

    current_counts = Counter()
    misplaced_alerts = []

    for box in results[0].boxes:
        raw_cls = model.names[int(box.cls[0])]
        if ignore_person and raw_cls == "person":
            continue

        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        disp_cls = to_vi_label(raw_cls, use_vi)
        current_counts[disp_cls] += 1

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        draw_label(annotated_img, f"{disp_cls} {conf_val:.2f}", (x1, max(y1 - 8, 20)))

        if cx < mid_x and cy < mid_y:
            zone = "Ô 1 (Trên-Trái)"
        elif cx >= mid_x and cy < mid_y:
            zone = "Ô 2 (Trên-Phải)"
        elif cx < mid_x and cy >= mid_y:
            zone = "Ô 3 (Dưới-Trái)"
        else:
            zone = "Ô 4 (Dưới-Phải)"

        cv2.circle(annotated_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        expected_raw = zone_rules[zone]
        if raw_cls != expected_raw:
            expected_disp = to_vi_label(expected_raw, use_vi)
            misplaced_alerts.append(
                f"⚠️ **{disp_cls}** nằm sai tại **{zone}** (Quy định: `{expected_disp}`)"
            )

    # Vẽ lưới 2x2 và nhãn từng ô
    cv2.line(annotated_img, (mid_x, 0), (mid_x, h), (0, 255, 255), 2)
    cv2.line(annotated_img, (0, mid_y), (w, mid_y), (0, 255, 255), 2)
    draw_label(annotated_img, f"O 1: {to_vi_label(zone_rules['Ô 1 (Trên-Trái)'], use_vi)}", (10, 25))
    draw_label(annotated_img, f"O 2: {to_vi_label(zone_rules['Ô 2 (Trên-Phải)'], use_vi)}", (mid_x + 10, 25))
    draw_label(annotated_img, f"O 3: {to_vi_label(zone_rules['Ô 3 (Dưới-Trái)'], use_vi)}", (10, mid_y + 25))
    draw_label(annotated_img, f"O 4: {to_vi_label(zone_rules['Ô 4 (Dưới-Phải)'], use_vi)}", (mid_x + 10, mid_y + 25))

    vn_now = get_vn_now()
    now_str = vn_now.strftime("%H:%M:%S")
    st.session_state.history.append({
        "Thời gian": now_str,
        "Tổng số lượng": sum(current_counts.values()),
        "Chi tiết": (
            ", ".join([f"{k}: {v}" for k, v in current_counts.items()])
            if current_counts
            else "Trống"
        ),
        "Số lỗi xếp sai": len(misplaced_alerts),
    })

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(
            annotated_img,
            caption="Bản đồ giám sát kệ hàng (Chấm đỏ = Tâm vật thể)",
            channels="BGR",
            use_container_width=True,
        )

    with col2:
        st.subheader("📊 Thống kê kiểm kê")
        if current_counts:
            for item, count in current_counts.items():
                st.write(f"- **{item}**: {count} cái")
        else:
            st.info("Chưa phát hiện hàng hóa nào.")

        st.divider()
        st.subheader("🚨 Cảnh báo vị trí")
        if misplaced_alerts:
            for alert in misplaced_alerts:
                st.error(alert)
        else:
            st.success("✅ Tất cả vật phẩm đều nằm đúng ô quy định.")

        st.divider()
        st.subheader("📉 Biến động kho")
        taken_items = []
        for item, prev_qty in st.session_state.prev_counts.items():
            curr_qty = current_counts.get(item, 0)
            if curr_qty < prev_qty:
                taken_items.append(
                    f"🔻 **{item}**: đã bị lấy đi **{prev_qty - curr_qty}** cái (còn {curr_qty})"
                )
        if taken_items:
            for msg in taken_items:
                st.warning(msg)
        else:
            st.write("Không có vật phẩm nào bị lấy đi.")

    st.session_state.prev_counts = current_counts

# --- BẢNG LỊCH SỬ & XUẤT BÁO CÁO ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Lịch sử kiểm kê & Xuất báo cáo")
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    csv_data = df_history.to_csv(index=False).encode("utf-8-sig")
    file_ts = get_vn_now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="⬇️ Tải báo cáo kiểm kê (.CSV)",
        data=csv_data,
        file_name=f"bao_cao_kho_{file_ts}.csv",
        mime="text/csv",
    )
