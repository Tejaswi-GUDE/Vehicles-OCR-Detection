import ssl
import re
import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
st.set_page_config(page_title="Vehicle Number Scanner", page_icon="🚗", layout="centered")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

INDIAN_PLATE_PATTERNS = [
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',      # MH12AB1234
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$', # Flexible common Indian pattern
    r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'            # 22BH1234AB
]

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def is_valid_indian_plate(text):
    return any(re.match(pattern, text) for pattern in INDIAN_PLATE_PATTERNS)

def normalize_common_misreads(text):
    chars = list(text)
    if len(chars) >= 4:
        for i in range(len(chars) - 4, len(chars)):
            if chars[i] == 'O':
                chars[i] = '0'
            elif chars[i] == 'I':
                chars[i] = '1'
            elif chars[i] == 'Z':
                chars[i] = '2'
            elif chars[i] == 'S':
                chars[i] = '5'
    return ''.join(chars)

def plate_score(text, conf):
    score = conf
    if is_valid_indian_plate(text):
        score += 3.0
    if 8 <= len(text) <= 10:
        score += 0.5
    return score

def detect_plate_regions(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)

    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, blackhat_kernel)

    grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    grad_x = np.absolute(grad_x)

    min_val, max_val = np.min(grad_x), np.max(grad_x)
    if max_val - min_val != 0:
        grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
    grad_x = grad_x.astype("uint8")

    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    h_img, w_img = img_rgb.shape[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect_ratio = w / float(h)

        if area < 1000:
            continue
        if not (2.0 <= aspect_ratio <= 6.5):
            continue
        if w < 60 or h < 20:
            continue

        pad_x = int(w * 0.08)
        pad_y = int(h * 0.15)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        crop = img_rgb[y1:y2, x1:x2]
        regions.append((crop, area))

    regions = sorted(regions, key=lambda r: r[1], reverse=True)
    return [r[0] for r in regions[:5]]

def generate_ocr_variants(crop_rgb):
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)

    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    enlarged_gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    enlarged_otsu = cv2.resize(otsu, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return [gray, otsu, adaptive, enlarged_gray, enlarged_otsu]

def extract_best_plate(img_rgb, reader):
    candidates = []

    plate_regions = detect_plate_regions(img_rgb)

    if not plate_regions:
        plate_regions = [img_rgb]

    for crop_rgb in plate_regions:
        variants = generate_ocr_variants(crop_rgb)

        for variant in variants:
            results = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                decoder='beamsearch',
                beamWidth=10,
                rotation_info=[90, 180, 270],
                contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.4,
                mag_ratio=1.5,
                canvas_size=3000
            )

            merged = ""
            conf_sum = 0.0
            count = 0

            for res in results:
                text = clean_text(res[1])
                conf = float(res[2])

                if len(text) < 2:
                    continue

                merged += text
                conf_sum += conf
                count += 1

                if len(text) >= 6:
                    candidates.append((text, conf))

                    corrected = normalize_common_misreads(text)
                    if corrected != text:
                        candidates.append((corrected, conf - 0.05))

            if merged and count > 0:
                avg_conf = conf_sum / count
                if len(merged) >= 6:
                    candidates.append((merged, avg_conf))

                    corrected_merged = normalize_common_misreads(merged)
                    if corrected_merged != merged:
                        candidates.append((corrected_merged, avg_conf - 0.05))

    if not candidates:
        return None

    unique_candidates = {}
    for text, conf in candidates:
        if text not in unique_candidates or conf > unique_candidates[text]:
            unique_candidates[text] = conf

    ranked = sorted(
        unique_candidates.items(),
        key=lambda x: plate_score(x[0], x[1]),
        reverse=True
    )

    for text, conf in ranked:
        if is_valid_indian_plate(text):
            return text

    return ranked[0][0] if ranked else None

st.title("🚗 Vehicle Number Scanner")
st.write("Upload a vehicle image or cropped number plate image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Scan Vehicle Number"):
        with st.spinner("Scanning with EasyOCR..."):
            plate_no = extract_best_plate(img_array, reader)

            if plate_no:
                st.success(f"Extracted Number: {plate_no}")

                if is_valid_indian_plate(plate_no):
                    st.info("Valid Indian number plate format ✅")
                else:
                    st.warning("Text detected, but format validation is weak ⚠️")
            else:
                st.error("No valid plate detected. Try a clearer or cropped image.")

st.markdown("---")
st.caption("Developed with Streamlit + EasyOCR")