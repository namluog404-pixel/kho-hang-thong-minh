from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import urlretrieve
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from numpy import asarray
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Kho hàng thông minh - Nhóm 5", layout="wide")
st.title("📦 Hệ thống Giám sát Kệ hàng Thông minh")


BEST_PT_URL = "https://github.com/Pluto235-RTX/Big5-RCUET/blob/main/TESTING/runs/my_project/obj_detecter/weights/best.pt"
LOCAL_WEIGHTS_PATH = Path("best.pt")

VI_LABELS = {
    "chocopie": "ChocoPie",
    "custas": "Custas",
    "milo": "Milo",
}

BASE_DIR = Path(__file__).resolve().parent
FONT_CANDIDATES = [
    BASE_DIR / "DejaVuSans-Bold.ttf",
    BASE_DIR / "DejaVuSans.ttf",
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
]


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
def load_font(size: int = 18):
    for font_path in FONT_CANDIDATES:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue
    return (
        ImageFont.load_default(size=size)
        if "size" in ImageFont.load_default.__code__.co_varnames
        else ImageFont.load_default()
    )


@st.cache_resource
def load_model(weights_url: str = BEST_PT_URL):
    if weights_url:
        cache_path = BASE_DIR / "downloaded_best.pt"
        if not cache_path.exists():
            urlretrieve(weights_url, cache_path)
        return YOLO(str(cache_path))
    if LOCAL_WEIGHTS_PATH.exists():
        return YOLO(str(LOCAL_WEIGHTS_PATH))
    return YOLO("yolov8n.pt")


def draw_label_pil(draw: ImageDraw.ImageDraw, text: str, pos: tuple[int, int], font):
    x, y = pos
    bbox = draw.textbbox((x, y), text, font=font)
    pad_x, pad_y = 5, 3
    draw.rectangle(
        [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y],
        fill=(0, 0, 0),
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


model = load_model()
label_font = load_font(size=18)
raw_classes = sorted(list(model.names.values()))


st.sidebar.header("⚙️ Cấu hình Kệ hàng (3 Ô Cạnh Nhau)")
lang_mode = st.sidebar.radio("Ngôn ngữ nhãn:", ("Tiếng Việt", "Tiếng Anh"), horizontal=True)
use_vi = lang_mode == "Tiếng Việt"

display_to_raw = {to_vi_label(c, use_vi): c for c in raw_classes}
display_classes = list(display_to_raw.keys())

default_raw = {
    "Ô 1 (Trái)": "chocopie" if "chocopie" in raw_classes else (raw_classes[0] if raw_classes else ""),
    "Ô 2 (Giữa)": "custas" if "custas" in raw_classes else (raw_classes[min(1, len(raw_classes) - 1)] if raw_classes else ""),
    "Ô 3 (Phải)": "milo" if "milo" in raw_classes else (raw_classes[min(2, len(raw_classes) - 1)] if raw_classes else ""),
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

if st.sidebar.button("🔄 Đặt lại lịch sử quét"):
    st.session_state.prev_counts = Counter()
    st.session_state.history = []
    st.sidebar.success("Đã làm mới toàn bộ dữ liệu.")


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

    pil_annotated = Image.fromarray(results[0].orig_img[:, :, ::-1].copy())
    draw = ImageDraw.Draw(pil_annotated)
    w, h = pil_annotated.size
    cut_1 = w // 3
    cut_2 = (2 * w) // 3

    current_counts = Counter()
    misplaced_alerts = []

    for box in results[0].boxes:
        raw_cls = model.names[int(box.cls[0])]
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        disp_cls = to_vi_label(raw_cls, use_vi)
        current_counts[disp_cls] += 1

        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
        draw_label_pil(draw, f"{disp_cls} {conf_val:.2f}", (x1, max(y1 - 24, 6)), label_font)

        if cx < cut_1:
            zone = "Ô 1 (Trái)"
        elif cx < cut_2:
            zone = "Ô 2 (Giữa)"
        else:
            zone = "Ô 3 (Phải)"

        r = 5
        draw.ellipse([int(cx) - r, int(cy) - r, int(cx) + r, int(cy) + r], fill=(255, 0, 0))

        expected_raw = zone_rules[zone]
        if raw_cls != expected_raw:
            expected_disp = to_vi_label(expected_raw, use_vi)
            misplaced_alerts.append(
                f"⚠️ **{disp_cls}** nằm sai tại **{zone}** (Quy định: `{expected_disp}`)"
            )

    draw.line([(cut_1, 0), (cut_1, h)], fill=(255, 255, 0), width=2)
    draw.line([(cut_2, 0), (cut_2, h)], fill=(255, 255, 0), width=2)
    draw_label_pil(draw, f"Ô 1: {to_vi_label(zone_rules['Ô 1 (Trái)'], use_vi)}", (10, 10), label_font)
    draw_label_pil(draw, f"Ô 2: {to_vi_label(zone_rules['Ô 2 (Giữa)'], use_vi)}", (cut_1 + 10, 10), label_font)
    draw_label_pil(draw, f"Ô 3: {to_vi_label(zone_rules['Ô 3 (Phải)'], use_vi)}", (cut_2 + 10, 10), label_font)

    annotated_img = asarray(pil_annotated)

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
            caption="Bản đồ giám sát kệ hàng 3 ô cạnh nhau (Chấm đỏ = Tâm vật thể)",
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
