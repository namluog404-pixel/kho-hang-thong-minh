from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from numpy import asarray
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Thanh toán thông minh - Nhóm 5", layout="wide")
st.title("🛒 Hệ thống Thanh toán Thông minh (AI Checkout)")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

PRODUCTS = {
    "chocopie": {"name": "Bánh ChocoPie (Hộp)", "default_price": 55000},
    "custas": {"name": "Bánh Custas (Hộp)", "default_price": 50000},
    "milo": {"name": "Sữa Milo (Hộp)", "default_price": 8000},
}

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
def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp mô hình tại '{model_path.name}'. "
            "Hãy tải best.pt lên cùng thư mục với app.py."
        )
    return YOLO(str(model_path))


def draw_label_pil(
    draw: ImageDraw.ImageDraw,
    text: str,
    pos: tuple[int, int],
    font,
    img_w: int,
    used_boxes: list[tuple[int, int, int, int]],
):
    x, y = pos
    pad_x, pad_y = 5, 3
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = max(pad_x, min(x, img_w - text_w - pad_x))
    y = max(pad_y, y)

    for _ in range(5):
        rect = (x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y)
        overlap = any(
            not (rect[2] < b[0] or rect[0] > b[2] or rect[3] < b[1] or rect[1] > b[3])
            for b in used_boxes
        )
        if not overlap:
            break
        y = max(pad_y, y - (text_h + pad_y * 2 + 2))

    final_rect = (x - pad_x, y - pad_y, x + text_w + pad_x, y + text_h + pad_y)
    used_boxes.append(final_rect)
    draw.rectangle(final_rect, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def build_vietqr_url(bank_id: str, account_no: str, account_name: str, amount: int, info: str) -> str:
    return (
        f"https://img.vietqr.io/image/{bank_id}-{account_no}-compact2.png"
        f"?amount={amount}&addInfo={quote(info)}&accountName={quote(account_name)}"
    )


model = load_model(MODEL_PATH)
label_font = load_font(size=18)

if "transactions" not in st.session_state:
    st.session_state.transactions = []

# --- SIDEBAR: CẤU HÌNH GIÁ & THANH TOÁN ---
st.sidebar.header("⚙️ Cấu hình Cửa hàng")
conf_threshold = st.sidebar.slider("Ngưỡng độ tự tin (Confidence)", 0.1, 1.0, 0.40, 0.05)
iou_threshold = st.sidebar.slider("Ngưỡng chồng lấn (IoU)", 0.1, 0.9, 0.45, 0.05)

st.sidebar.subheader("💰 Bảng đơn giá (VNĐ)")
prices = {}
for key, meta in PRODUCTS.items():
    prices[key] = st.sidebar.number_input(
        meta["name"], min_value=1000, step=1000, value=meta["default_price"]
    )

st.sidebar.divider()
st.sidebar.subheader("🏦 Thông tin VietQR (Demo)")
bank_id = st.sidebar.selectbox("Ngân hàng", ["MB", "VCB", "TCB", "BIDV", "ICB", "ACB", "TPB"])
account_no = st.sidebar.text_input("Số tài khoản", value="0123456789")
account_name = st.sidebar.text_input("Tên chủ tài khoản", value="NHOM 5 UET")

if st.sidebar.button("🗑️ Xóa lịch sử giao dịch"):
    st.session_state.transactions = []
    st.sidebar.success("Đã làm mới lịch sử giao dịch.")

# --- KHU VỰC QUÉT KHAY HÀNG ---
input_option = st.radio("Nguồn ảnh khay thanh toán:", ("Chụp từ Camera", "Tải ảnh lên"), horizontal=True)
img_file = (
    st.camera_input("Đặt sản phẩm dàn trải trên khay và chụp")
    if input_option == "Chụp từ Camera"
    else st.file_uploader("Chọn ảnh khay hàng...", type=["jpg", "jpeg", "png"])
)

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    image.thumbnail((800, 800))

    with st.spinner("Đang nhận diện sản phẩm trên khay bằng YOLO..."):
        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

    pil_annotated = image.copy()
    draw = ImageDraw.Draw(pil_annotated)
    w, _ = pil_annotated.size

    cart_counts = Counter()
    used_label_boxes = []

    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            conf_val = float(box.conf[0])
            cls_id = int(box.cls[0])
            raw_cls = str(model.names[cls_id]).strip().lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cart_counts[raw_cls] += 1
            disp_name = PRODUCTS.get(raw_cls, {}).get("name", raw_cls).split(" (")[0]

            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
            draw_label_pil(
                draw,
                f"{disp_name} {conf_val:.2f}",
                (x1, max(y1 - 24, 6)),
                label_font,
                w,
                used_label_boxes,
            )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(asarray(pil_annotated), caption="Khay hàng đã nhận diện", use_container_width=True)

    with col2:
        st.subheader("🧾 Hóa đơn thanh toán")
        if not cart_counts:
            st.info("Chưa phát hiện sản phẩm nào trên khay.")
        else:
            rows = []
            total_amount = 0
            for raw_cls, qty in cart_counts.items():
                unit_price = prices.get(raw_cls, 0)
                subtotal = qty * unit_price
                total_amount += subtotal
                rows.append({
                    "Sản phẩm": PRODUCTS.get(raw_cls, {}).get("name", raw_cls),
                    "SL": qty,
                    "Đơn giá": f"{unit_price:,} đ",
                    "Thành tiền": f"{subtotal:,} đ",
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.metric(label="Tổng cộng cần thanh toán", value=f"{total_amount:,} VNĐ")

            order_code = f"DH{get_vn_now().strftime('%H%M%S')}"
            qr_url = build_vietqr_url(bank_id, account_no, account_name, total_amount, f"Thanh toan {order_code}")
            st.image(qr_url, caption=f"Quét VietQR để thanh toán ({order_code})", width=260)

            if st.button("✅ Xác nhận đã thanh toán", type="primary", use_container_width=True):
                st.session_state.transactions.append({
                    "Mã đơn": order_code,
                    "Thời gian": get_vn_now().strftime("%H:%M:%S"),
                    "Chi tiết": ", ".join(
                        f"{PRODUCTS.get(k, {}).get('name', k).split(' (')[0]} x{v}"
                        for k, v in cart_counts.items()
                    ),
                    "Tổng tiền (VNĐ)": total_amount,
                })
                st.success(f"Đã ghi nhận giao dịch {order_code}.")

# --- LỊCH SỬ GIAO DỊCH ---
if st.session_state.transactions:
    st.divider()
    st.subheader("📋 Lịch sử giao dịch & Doanh thu")
    df_tx = pd.DataFrame(st.session_state.transactions)
    st.dataframe(df_tx, use_container_width=True, hide_index=True)

    total_rev = df_tx["Tổng tiền (VNĐ)"].sum()
    st.write(f"**Tổng doanh thu phiên làm việc:** {total_rev:,} VNĐ")

    csv_data = df_tx.to_csv(index=False).encode("utf-8-sig")
    file_ts = get_vn_now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="⬇️ Xuất báo cáo doanh thu (.CSV)",
        data=csv_data,
        file_name=f"doanh_thu_{file_ts}.csv",
        mime="text/csv",
    )
