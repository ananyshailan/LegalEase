import streamlit as st
import pytesseract
from fpdf import FPDF
import cv2
import tempfile
import os

# Function to convert image to text using OCR
def convert_image_to_text(image_path, min_confidence=0.6):
    # Read the image
    img = cv2.imread(image_path)

    # Preprocess the image (optional)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray)

    # Perform OCR on the preprocessed image
    custom_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_data(enhanced_img if enhanced_img is not None else img, config=custom_config, output_type=pytesseract.Output.DICT)

    # Extract text with confidence score filtering
    text = ' '.join([result['text'][i] for i in range(len(result['text'])) if int(result['conf'][i]) >= min_confidence * 100])
    return text

# Function to convert text to PDF
# Function to convert text to PDF without custom font
# def convert_text_to_pdf(text, pdf_file):
#     # Create instance of FPDF class
#     pdf = FPDF()

#     # Add a page
#     pdf.add_page()

#     # Use default font (no need to add a custom font)
#     pdf.set_font('Arial', size=12)

#     # Split text into lines with a maximum of 95 characters each
#     lines = [text[i:i+95] for i in range(0, len(text), 95)]

#     # Add each line to the PDF
#     for line in lines:
#         pdf.cell(200, 8, txt=line, ln=True, align='L')

#     # Save the PDF
#     pdf.output(pdf_file)
# Function to convert text to PDF without custom font, with UTF-8 encoding
def convert_text_to_pdf(text, pdf_file):
    # Create instance of FPDF class
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Use default font (no need to add a custom font)
    pdf.set_font('Arial', size=12)

    # Split text into lines with a maximum of 95 characters each
    lines = [text[i:i+95] for i in range(0, len(text), 95)]

    # Add each line to the PDF
    for line in lines:
        # Encode line to 'latin-1', replacing unsupported characters with '?'
        pdf.cell(200, 8, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=True, align='L')

    # Save the PDF
    pdf.output(pdf_file)



# Streamlit Frontend
st.title("Scan your documents")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_image.read())
        image_path = temp_file.name

    # Display the uploaded image
    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Extract text from the image
    text = convert_image_to_text(image_path)
    st.subheader("Extracted Text")
    st.text_area("Text", text, height=200)

    # Button to convert text to PDF
    if st.button("Convert to PDF"):
        pdf_file = "extracted_text.pdf"
        convert_text_to_pdf(text, pdf_file)

        # Provide download link for the PDF
        with open(pdf_file, "rb") as file:
            st.download_button(
                label="Download PDF",
                data=file,
                file_name="extracted_text.pdf",
                mime="application/pdf"
            )

    # Clean up temporary files
    os.remove(image_path)
