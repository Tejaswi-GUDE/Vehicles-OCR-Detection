import os
import re
import cv2
import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

INDIAN_PLATE_PATTERNS = [
    r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$',
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$',
    r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$'
]

def clean_text(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(text) > 10:
        text = text[:10]
    return text

def is_valid_indian_plate(text):
    return any(re.match(pattern, text) for pattern in INDIAN_PLATE_PATTERNS)

def normalize_common_misreads(text):
    replacements = {
        'O': '0', 'I': '1', 'Z': '2',
        'S': '5', 'B': '8'
    }

    corrected = ""
    for i, ch in enumerate(text):
        if i >= len(text) - 4:
            corrected += replacements.get(ch, ch)
        else:
            corrected += ch

    return corrected

def plate_score(text, conf):
    score = conf
    if is_valid_indian_plate(text):
        score += 3.0
    if 8 <= len(text) <= 10:
        score += 0.5
    return score

def detect_plate_regions(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
    h_img, w_img = img_bgr.shape[:2]

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

        mean_intensity = np.mean(gray[y:y+h, x:x+w])
        if mean_intensity < 50 or mean_intensity > 200:
            continue

        pad_x = int(w * 0.08)
        pad_y = int(h * 0.15)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        crop = img_bgr[y1:y2, x1:x2]
        regions.append((crop, area))

    regions = sorted(regions, key=lambda r: r[1], reverse=True)
    return [r[0] for r in regions[:5]]

def generate_ocr_variants(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    blur = cv2.bilateralFilter(sharpened, 11, 17, 17)

    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    resized = cv2.resize(blur, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    return [blur, otsu, adaptive, resized]

def extract_vehicle_number(image_path):
    if not os.path.exists(image_path):
        return "Error: Image file not found!"

    img = cv2.imread(image_path)
    if img is None:
        return "Error: Unable to read image!"

    img = cv2.resize(img, (640, 480))

    candidates = []
    plate_regions = detect_plate_regions(img)

    if not plate_regions:
        plate_regions = [img]

    for crop in plate_regions:
        variants = generate_ocr_variants(crop)

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
                    corrected_merged = normalize_common_misreads(merged)
                    if corrected_merged != merged:
                        candidates.append((corrected_merged, avg_conf - 0.05))

    if not candidates:
        return "No valid plate detected"

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

    return ranked[0][0] if ranked else "No valid plate detected"


if __name__ == "__main__":
    test_image = "test_plate.jpg"
    print(f"Processing: {test_image} ...")
    result = extract_vehicle_number(test_image)
    print(f"Final Output: {result}")
