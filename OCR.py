import ssl
import re
import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context
st.set_page_config(page_title="Vehicle Number Scanner", page_icon="🚗")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# -------- SAME FUNCTIONS AS ABOVE --------
# (Copy ALL functions from app.py except extract_vehicle_number)
# Replace only extract function below

def extract_best_plate(img_rgb, reader):
    img_rgb = cv2.resize(img_rgb, (640, 480))

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
                beamWidth=5,
                rotation_info=[0],
                text_threshold=0.7,
                low_text=0.4,
                link_threshold=0.5,
                mag_ratio=1.8,
                canvas_size=2560
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
        if conf < 0.4:
            continue
        if is_valid_indian_plate(text):
            return text

    return ranked[0][0] if ranked else None


# -------- UI --------

st.title("🚗 Vehicle Number Scanner")
st.write("Upload a vehicle image or cropped number plate image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Scan Vehicle Number"):
        with st.spinner("Scanning..."):
            plate_no = extract_best_plate(img_array, reader)

            if plate_no:
                st.success(f"Extracted Number: {plate_no}")
            else:
                st.error("No valid plate detected.")
