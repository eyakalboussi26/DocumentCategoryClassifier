import os
import tempfile
import streamlit as st
import cv2
import fitz  # PyMuPDF
from ultralytics import YOLO
import pandas as pd

# CSS for styling
st.markdown("""
    <style>
        .stButton > button {
            background-color: #b9b0a4;
            border: 2px solid #937a5a;
            border-radius: 10px;
            box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.3);
            color: white;
            font-family: 'Times New Roman', Times, serif;
            font-size: 16px;
            padding: 10px 20px;
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #937a5a;
            border: 2px solid #b9b0a4;
            box-shadow: 3px 3px 12px rgba(0, 0, 0, 0.4);
            transform: translateY(-2px);
        }
        .image-carousel-title, .loading-text, .section-title {
            color: #4b2e12;
            font-family: 'Times New Roman', Times, serif;
            font-weight: bold;
            text-align: center;
        }
        .image-carousel-title {
            font-size: 22px;
        }
        .loading-text {
            font-size: 18px;
            padding: 10px 0;
        }
        .section-title {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .section-bg-1 {
            background-color: #ddbd96;
            padding: 20px;
            border-radius: 10px;
        }
        .section-bg-2 {
            background-color: #f5f0e6;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with project description
st.sidebar.title("Welcome to Our Web Application")
st.sidebar.markdown("""
    ### Project Description
    This project converts PDF pages to images and applies YOLO classification on them.
""")

# Define navigation pages
def main_page():
    st.markdown("<div class='section-bg-1'>", unsafe_allow_html=True)
    st.markdown("<h1 class='section-title'>PDF to Image Conversion and YOLOv5 Classification</h1>", unsafe_allow_html=True)

    # Step 1: Upload PDF section
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")
    st.markdown("</div>", unsafe_allow_html=True)

    # Step 2: PDF to Images Section
    st.markdown("<div class='section-bg-2'>", unsafe_allow_html=True)

    if uploaded_pdf:
        with st.spinner("Loading PDF... Please wait"):
            st.markdown("<p class='loading-text'>Loading PDF...</p>", unsafe_allow_html=True)

        # Temporary directory for storing images
        temp_dir = tempfile.mkdtemp()

        # Convert PDF to images
        with st.spinner("Converting PDF to images... Please wait"):
            st.markdown("<p class='loading-text'>Converting PDF to images...</p>", unsafe_allow_html=True)
            pdf_document = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
            images = []
            detection_log = []  # To store classification information

            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                output_image_path = os.path.join(temp_dir, f'page_{page_num + 1}.png')
                pix.save(output_image_path)
                images.append(output_image_path)

            pdf_document.close()

        st.success("PDF conversion completed!")

        # YOLOv5 Classification Section
        st.markdown("<h2 class='image-carousel-title'>Classifying Images with YOLOv5</h2>", unsafe_allow_html=True)

        model_path = 'C:/Users/ekalboussi/OneDrive - ALTEN Group/Bureau/Project_NLP/OCR-and-Classifier-Doc/Classification/src/trained_models/trained-model.pt'
        model = YOLO(model_path)

        modified_images = []  # To store modified images with predictions

        with st.spinner("Classifying images... Please wait"):
            st.markdown("<p class='loading-text'>Classifying images...</p>", unsafe_allow_html=True)

            for img_path in images:
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (640, 640))
                results = model(img_resized, conf=0.5, imgsz=640)  # Ensure correct input size and confidence

                for result in results:
                    for box in result.boxes:
                        xyxy = box.xyxy.cpu().numpy().flatten()
                        conf = box.conf.cpu().numpy().item()
                        class_name = result.names[int(box.cls)]
                        detection_log.append({
                            "Page": os.path.basename(img_path),
                            "Class": class_name,
                            "Confidence": conf,
                            "Coordinates": xyxy.tolist()
                        })

                        # Draw bounding box and label on the image
                        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(img, f"{class_name}: {conf*100:.2f}%", 
                                    (int(xyxy[0]), int(xyxy[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Save the modified image with predictions
                modified_image_path = os.path.join(temp_dir, f'predicted_{os.path.basename(img_path)}')
                cv2.imwrite(modified_image_path, img)  # Save the image with bounding boxes
                modified_images.append(modified_image_path)  # Append modified image for display

        st.success("Classification complete!")

        # Store results in session state for use in other pages
        st.session_state.detection_log = detection_log
        st.session_state.modified_images = modified_images
        st.session_state.pred_index = 0  # Initialize the image index for the carousel
        st.session_state.page = "main_page"  # Stay on the main page

        # Buttons for showing results
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Show Classification Log"):
                st.session_state.page = "log_page"  # Change page to log page

        with col2:
            if st.button("Show Predicted Images"):
                st.session_state.page = "predicted_page"  # Change page to predicted images page

    st.markdown("</div>", unsafe_allow_html=True)  # Close the section

def log_page():
    st.markdown("<div class='section-bg-1'>", unsafe_allow_html=True)
    st.markdown("<h2 class='image-carousel-title'>Classification Log</h2>", unsafe_allow_html=True)

    if "detection_log" in st.session_state:
        df_log = pd.DataFrame(st.session_state.detection_log)
        st.dataframe(df_log)

    if st.button("Back to Main Page"):
        st.session_state.page = "main_page"  # Change back to main page

    st.markdown("</div>", unsafe_allow_html=True)

def predicted_page():
    st.markdown("<div class='section-bg-1'>", unsafe_allow_html=True)
    st.markdown("<h2 class='image-carousel-title'>Predicted Images</h2>", unsafe_allow_html=True)

    if "modified_images" in st.session_state:
        num_images = len(st.session_state.modified_images)

        # Display the current predicted image based on the index
        current_image = st.session_state.modified_images[st.session_state.pred_index]
        st.image(current_image, use_column_width=True)

        # Navigation buttons for the carousel
        col_left, col_right = st.columns(2)

        with col_left:
            if st.button("⬅️ Previous"):
                if st.session_state.pred_index > 0:
                    st.session_state.pred_index -= 1

        with col_right:
            if st.button("Next ➡️"):
                if st.session_state.pred_index < num_images - 1:
                    st.session_state.pred_index += 1

        st.write(f"Image {st.session_state.pred_index + 1} of {num_images}")

    if st.button("Back to Main Page"):
        st.session_state.page = "main_page"  # Change back to main page

    st.markdown("</div>", unsafe_allow_html=True)

# Navigation between pages
if "page" not in st.session_state:
    st.session_state.page = "main_page"

# Display the current page based on session state
if st.session_state.page == "main_page":
    main_page()
elif st.session_state.page == "log_page":
    log_page()
elif st.session_state.page == "predicted_page":
    predicted_page()
