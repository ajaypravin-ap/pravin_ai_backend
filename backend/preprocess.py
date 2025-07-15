import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import cv2
import pytesseract

def extract_numbers_and_features(image_path):
    print("📷 Loading image from:", image_path)
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found or unreadable.")
        return []

    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run OCR
    text = pytesseract.image_to_string(gray)
    print("🔍 OCR Text:", text)

    # Extract digits only
    digits = [int(char) for char in text if char.isdigit()]
    print("🔢 Extracted Digits:", digits)

    results = []
    for digit in digits:
        size = "Big" if digit >= 5 else "Small"
        color = "Red" if digit in [0, 2, 4, 6, 8] else "Green"
        results.append({
            "number": digit,
            "size": size,
            "color": color
        })

    print("✅ Final extracted data:", results)
    return results