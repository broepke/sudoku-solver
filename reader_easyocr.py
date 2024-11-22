reader = easyocr.Reader(['en'])
ocr_result = reader.readtext(preprocessed_image, detail=1)
print("OCR Results:", ocr_result)