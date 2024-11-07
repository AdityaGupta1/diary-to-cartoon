import env
from google.cloud import vision
from PIL import Image, ImageEnhance, ImageOps
from io import BytesIO
import cv2
import numpy as np


def extract_text_as_paragraph():
    # Setting up the Google Vision API 
    client = vision.ImageAnnotatorClient.from_service_account_file(env.GOOGLE_VISION_API_KEY_PATH)
    
    # Open and preprocess the image
    with Image.open("testImages/raw_diary_japanesegirl_eng.png") as img:
        # Correct orientation based on EXIF metadata
        img = ImageOps.exif_transpose(img)
        
        # Convert to grayscale
        grey_scale_image = img.convert('L')
        
        # Step 1: Increase contrast to make text stand out
        enhancer = ImageEnhance.Contrast(grey_scale_image)
        enhanced_image = enhancer.enhance(2.2)
        enhanced_image.save("./preprocessed_images/debug_contrast_enhanced.jpg")  # Save for debugging

        # Convert to OpenCV format for further processing
        image_cv = np.array(enhanced_image)

        # Step 2: Apply Bilateral Filter to reduce noise while preserving edges
        denoised_image = cv2.bilateralFilter(image_cv, 9, 75, 75)
        Image.fromarray(denoised_image).save("./preprocessed_images/debug_denoised_bilateral.jpg")  # Save for debugging

        # Step 3: Apply Median Blur to remove small noise artifacts further
        denoised_image = cv2.medianBlur(denoised_image, 3)
        Image.fromarray(denoised_image).save("./preprocessed_images/debug_denoised_median.jpg")  # Save for debugging

        # Step 5: Apply sharpening to make text edges clearer
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_image = cv2.filter2D(denoised_image, -1, sharpen_kernel)
        preprocessed_image = Image.fromarray(sharpened_image)
        preprocessed_image.save("./preprocessed_images/preprocessed_debug_image.jpg")  # Final preprocessed image for OCR

        # Convert processed image to bytes for OCR
        img_byte_arr = BytesIO()
        preprocessed_image.save(img_byte_arr, format="JPEG")
        content = img_byte_arr.getvalue()
    
    # Send the image to Google Vision API for OCR
    image = vision.Image(content=content)
    
    # Handle the response with error checking
    try:
        response = client.document_text_detection(image=image)
        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")
    except Exception as e:
        print(e)
        return ""
    
    # Extract text and format it as a continuous passage
    if not response.text_annotations:
        return ""
    
    diary_contents = response.text_annotations[0].description
    diary_contents = diary_contents.replace('\n', ' ')  # Remove line breaks
    
    return diary_contents

# Call the function and print the result
content = extract_text_as_paragraph()
print(content)
