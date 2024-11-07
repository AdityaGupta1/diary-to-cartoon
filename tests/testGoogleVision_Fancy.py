import env
import os
from google.cloud import vision
from PIL import Image, ImageEnhance, ImageOps
from io import BytesIO
import cv2
import numpy as np
from difflib import SequenceMatcher
import google.generativeai as genai
genai.configure(api_key=env.GEMINI_API_KEY)


'''
This file is used to test the validity of OCR tools used in this project
Cross-validation, multiple pipelines are utlized
The generative result looks really promising
'''

def preprocess_pipeline_01(img):
    '''
    Applies:
    1. GreyScale
    2. Contrast enhancement
    3. Bilateral Filter to reduce Noise
    4. Apply Median Blur to remove small noise artifacts
    5. Sharpening
    '''
    # Create the debug directory if it doesn't exist
    debug_dir = './debugs/pipeline01_debug'
    os.makedirs(debug_dir, exist_ok=True)

    # Step 1: Convert to grayscale and enhance contrast
    grey_scale_img = img.convert('L')
    enhancer = ImageEnhance.Contrast(grey_scale_img)
    enhanced_img = enhancer.enhance(2.2)
    enhanced_img.save(f'{debug_dir}/step01_enhanced_debug.jpg')

    # Convert to OpenCV format
    image_cv = np.array(enhanced_img)

    # Step 2: Apply Bilateral Filter to reduce noise while preserving edges
    denoised_image = cv2.bilateralFilter(image_cv, 9, 75, 75)
    Image.fromarray(denoised_image).save(f"{debug_dir}/step02_denoised_bilateral_debug.jpg")

    # Step 3: Apply Median Blur to further remove small noise artifacts
    denoised_image = cv2.medianBlur(denoised_image, 3)
    Image.fromarray(denoised_image).save(f"{debug_dir}/step03_denoised_median_debug.jpg")

    # Step 4: Apply Sharpening to make text edges cleaner
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(denoised_image, -1, sharpen_kernel)
    preprocessed_image = Image.fromarray(sharpened_image)
    preprocessed_image.save(f"{debug_dir}/step04_sharpen_img_debug.jpg")  # Final preprocessed image for OCR
    
    return preprocessed_image

def preprocess_pipeline_02(img):
    '''
    Applies:
    1. High Contrast Adjustment
    2. CLAHE for Local Contrast Enhancement
    3. Noise Reduction with Non-Local Means
    4. Median Blur for Small Noise Reduction
    5. Adaptive Thresholding for Binarization
    6. Morphological Opening to Remove Small Noise Artifacts
    '''
    
    # Create the debug directory if it doesn't exist
    debug_dir = './debugs/pipeline02_debug'
    os.makedirs(debug_dir, exist_ok=True)

    # Convert to grayscale
    grey_scale_img = img.convert('L')
    Image.fromarray(np.array(grey_scale_img)).save(f"{debug_dir}/step00_GrayScale_debug.jpg")
    
    # Step 1: High contrast adjustment
    enhancer = ImageEnhance.Contrast(grey_scale_img)
    high_contrast_img = enhancer.enhance(2.2)
    Image.fromarray(np.array(high_contrast_img)).save(f"{debug_dir}/step01_HighContrast_debug.jpg")

    # Step 2: CLAHE for localized contrast enhancement
    image_cv = np.array(high_contrast_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image_cv)
    Image.fromarray(clahe_image).save(f"{debug_dir}/step02_CLAHE_debug.jpg")

    # Step 3: Noise Reduction with Non-Local Means
    denoised_image = cv2.fastNlMeansDenoising(clahe_image, None, h=30, templateWindowSize=7, searchWindowSize=21)
    Image.fromarray(denoised_image).save(f"{debug_dir}/step03_NonLocalMeansDenoising_debug.jpg")

    # Step 4: Apply Median Blur for additional small noise reduction
    median_blurred_image = cv2.medianBlur(denoised_image, 3)
    Image.fromarray(median_blurred_image).save(f"{debug_dir}/step04_MedianBlur_debug.jpg")

    # Step 5: Apply Adaptive Thresholding for binarization
    adaptive_thresh = cv2.adaptiveThreshold(
        median_blurred_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    Image.fromarray(adaptive_thresh).save(f"{debug_dir}/step05_AdaptiveThreshold_debug.jpg")
    # Convert final processed image to PIL format
    preprocessed_image = Image.fromarray(adaptive_thresh)
    return preprocessed_image

def preprocess_pipeline_03(img):
    # Pipeline 2: Adaptive thresholding for variable lighting
    debug_dir = './debugs/pipeline03_debug'
    os.makedirs(debug_dir, exist_ok=True)
    img = img.convert('L')
    image_cv = np.array(img)
    image_cv = cv2.medianBlur(image_cv, 3)
    adaptive_thresh = cv2.adaptiveThreshold(
        image_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    preprocessed_image = Image.fromarray(adaptive_thresh)
    preprocessed_image.save(f"{debug_dir}/preprocessed_debug.jpg")
    return preprocessed_image
    


def ocr_image(image_bytes):
    '''
    Performs OCR on the given image bytes and returns the extracted text
    '''
    client = vision.ImageAnnotatorClient.from_service_account_file(env.GOOGLE_VISION_API_KEY_PATH)
    image = vision.Image(content=image_bytes)
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
    
    return diary_contents  # Text representation of the diary


    
def extract_text_and_consolidate():
    # Load image and correct orientation
    with Image.open("testImages/raw_diary_japanesegirl_eng.png") as diary_image:
        img = ImageOps.exif_transpose(diary_image)  # Transpose the image based on its EXIF information
    
    # Apply preprocessing pipeline
    preprocessed_images = [
        preprocess_pipeline_01(img),
        preprocess_pipeline_02(img),
        preprocess_pipeline_03(img),
    ]
    ocr_texts = []  # Store the texts content returned by Google Vision OCR Tools
    
    # Run OCR on the processed images
    for i, processed_img in enumerate(preprocessed_images):
        img_byte_arr = BytesIO()
        processed_img.save(img_byte_arr, format="JPEG")  # Fixed typo here
        content = img_byte_arr.getvalue()
        text = ocr_image(content)
        ocr_texts.append(text)
    

    # Validate and correct text using Gemini
    # final_text = validate_and_correct_text(consolidated_text)

    return ocr_texts, img  # Return the OCR texts and original image

# Let the LLM create tailored contents
def validate_and_consolidate_with_gemini(ocr_texts, img):
    """
    Consolidates multiple OCR results and validates them with the original image using Gemini.
    """
    # Convert the image to bytes to send to Gemini
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    # Prepare the concatenated OCR text for Gemini
    ocr_texts_concatenated = "\n\n".join([f"Version {i+1}:\n{text}" for i, text in enumerate(ocr_texts)])

    # Craft the prompt to guide Gemini
    prompt = f"""
    Below are multiple OCR-generated text versions extracted from the same image. Each version may contain errors or inconsistencies. Please analyze these versions and the attached image to produce a single, coherent, and semantically accurate text that accurately reflects the content in the image.

    OCR Text Versions:
    {ocr_texts_concatenated}
    
    Refer to the image for context and correct any OCR errors, inconsistencies, or logical gaps. Ensure the final output is readable and logical, preserving the original meaning as depicted in the image.
    """

    # Send the prompt along with the image to Gemini
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(
        [prompt, img],
    )

    # Retrieve and return the corrected text
    if response:
        corrected_text = response.text.strip()
        return corrected_text
    else:
        return "Error: Could not retrieve consolidated text from Gemini."

# test
ocr_texts, original_img = extract_text_and_consolidate() 
final_corrected_text = validate_and_consolidate_with_gemini(ocr_texts, original_img)
print(final_corrected_text)