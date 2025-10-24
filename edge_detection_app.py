import streamlit as st
import cv2
import numpy as np
from PIL import Image

# setting page configuration
st.set_page_config(
    page_title="Interactive Edge Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Light Mode Theme Colors */
    :root {
        --primary-bg: #f0f2f6;      /* Very light gray background */
        --secondary-bg: #ffffff;    /* Pure white for containers/cards */
        --text-color: #1a1a1a;      /* Dark text */
        --highlight-color: #0078d4; /* A professional blue highlight */
        --border-color: #dddddd;    /* Light gray border */
    }

    /* Main container styling for a light mode look */
    .stApp {
        background-color: var(--primary-bg);
        color: var(--text-color);
    }
    
    /* Header and Title styling */
    .st-emotion-cache-1jm15v3, .st-emotion-cache-12wi5q4 {
        color: var(--highlight-color); 
        text-shadow: none; /* Removed dark mode shadow */
    }
    
    /* Customize the sidebar title and background */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-bg);
        color: var(--text-color);
        border-right: 1px solid var(--border-color);
    }
    
    /* Main titles for Input/Output images */
    .image-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-color);
        text-align: center;
        margin-bottom: 10px;
        padding: 5px;
        border-bottom: 2px solid var(--border-color);
    }
    
    /* Style for the image columns (cards) */
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        padding: 15px;
        border-radius: 10px;
        background-color: var(--secondary-bg);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* Lighter shadow for light mode */
    }
    
    /* Custom radio buttons box in sidebar */
    [data-testid="stForm"] > [data-testid="stVerticalBlock"] {
        background-color: #f7f9fc; /* Slightly darker than secondary-bg for contrast */
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid var(--border-color);
    }
    
    /* Ensure image display is clean */
    img {
        border-radius: 8px;
        border: 2px solid var(--border-color);
    }
    
    /* Input widgets (sliders, selectbox text) */
    .st-emotion-cache-16niy0n {
        color: var(--text-color);
    }

</style>
""", unsafe_allow_html=True)

st.title("Edge Detection Playground")
st.markdown("### Interactive Visual Experimentation with Computer Vision Algorithms")

# adding edge detection function: 

def canny_detector(img, lower_threshold, upper_threshold, kernel_size, sigma):
    """Applies Canny edge detection after Gaussian blur."""
  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # applying Gaussian Blur 
    ksize = int(kernel_size)
    if ksize % 2 == 0:
        ksize += 1
    
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    
    # Apply Canny
    edges = cv2.Canny(blur, lower_threshold, upper_threshold)
    

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def sobel_detector(img, kernel_size, direction):
    """Applies Sobel edge detection in X, Y, or both directions."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if direction == "X-Gradient":
        # calculating gradient in X direction
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        result = abs_grad_x
    elif direction == "Y-Gradient":
        # calculating gradient in Y direction
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        result = abs_grad_y
    else: # Both X and Y (Magnitude)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        
        # Calculate total gradient magnitude
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        result = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def laplacian_detector(img, kernel_size):
    """Applies Laplacian edge detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
    
    result = cv2.convertScaleAbs(laplacian)
    
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

# adding different parameters
with st.sidebar:
    st.header("⚙️ Algorithm Selection & Parameters")
    
    # for selecting detection method
    algorithm = st.radio(
        "Select Edge Detection Algorithm:",
        ["Canny", "Sobel", "Laplacian"],
        index=0,
        key="selected_algo"
    )
    
    st.markdown("---")
    st.subheader(f"Parameters for: {algorithm}")

    # variation of parameters according to detection method
    if algorithm == "Canny":
        st.markdown("The most robust and widely used edge detector.")
        st.markdown("**Gaussian Blur:** Controls noise and detail level.")
        sigma = st.slider("Sigma (Gaussian Blur)", 0.0, 5.0, 1.0, 0.1, key="canny_sigma")
        kernel_size = st.selectbox("Kernel Size (Gaussian Blur)", [3, 5, 7, 9], index=1, key="canny_ksize")
        st.markdown("**Thresholds:** Defines which gradients are considered edges.")
        lower_threshold = st.slider("Lower Threshold (T1)", 0, 255, 50, 5, key="canny_t1")
        upper_threshold = st.slider("Upper Threshold (T2)", 0, 255, 150, 5, key="canny_t2")
        
    
        if upper_threshold <= lower_threshold:
            st.warning("Upper Threshold (T2) must be greater than Lower Threshold (T1)!")

    elif algorithm == "Sobel":
        st.markdown("Gradient-based detector, sensitive to noise.")
        kernel_size = st.selectbox("Kernel Size (Aperture)", [3, 5, 7], index=0, key="sobel_ksize")
        direction = st.radio(
            "Gradient Direction:",
            ["Both X and Y (Magnitude)", "X-Gradient", "Y-Gradient"],
            index=0,
            key="sobel_direction"
        )
        
    elif algorithm == "Laplacian":
        st.markdown("Second-order derivative operator, often highlights noisy details.")
        kernel_size = st.selectbox("Kernel Size (Aperture)", [1, 3, 5], index=1, key="lap_ksize")

# uploading and displaying image

uploaded_file = st.file_uploader(
    "Upload an Image (JPG, PNG, BMP)", 
    type=["jpg", "jpeg", "png", "bmp"],
    help="Select an image from your local file system to begin edge detection."
)

# input and output display
col1, col2 = st.columns(2)

if uploaded_file is not None:

    try:
      
        pil_img = Image.open(uploaded_file)
        img_array = np.array(pil_img)
        original_img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Display Input Image:
        with col1:
            st.markdown('<p class="image-title">Input: Original Image</p>', unsafe_allow_html=True)
            st.image(original_img_bgr, channels="BGR", caption=f"Original Image: {uploaded_file.name}", use_container_width=True)

        # --- Image Processing & Output Display ---
        processed_img_bgr = None

        if algorithm == "Canny":
            if upper_threshold > lower_threshold:
                processed_img_bgr = canny_detector(
                    original_img_bgr, 
                    lower_threshold, 
                    upper_threshold, 
                    kernel_size, 
                    sigma
                )
            else:

                processed_img_bgr = original_img_bgr
                
        elif algorithm == "Sobel":
            processed_img_bgr = sobel_detector(
                original_img_bgr, 
                kernel_size, 
                direction
            )
            
        elif algorithm == "Laplacian":
            processed_img_bgr = laplacian_detector(
                original_img_bgr, 
                kernel_size
            )
        
        # Display Output Image
        with col2:
            st.markdown('<p class="image-title">Output: Edge Detected Image</p>', unsafe_allow_html=True)
            if processed_img_bgr is not None:
       
                st.image(processed_img_bgr, channels="BGR", caption=f"Result using {algorithm} Algorithm", use_container_width=True)
            else:

                 st.image(original_img_bgr, channels="BGR", caption="Adjust Canny Thresholds to run the algorithm.", use_container_width=True)
                 
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.warning("Please ensure the uploaded file is a valid image format.")

else:
    # Placeholder when no image is uploaded
    st.info("⬆️ Please upload an image using the file uploader above to begin edge detection.")
    # Placeholder using light mode colors
    with col1:
        st.markdown('<p class="image-title">Input: Original Image</p>', unsafe_allow_html=True)

        st.image("https://placehold.co/600x400/f0f2f6/1a1a1a?text=Upload+Image", use_container_width=True)
    with col2:
        st.markdown('<p class="image-title">Output: Edge Detected Image</p>', unsafe_allow_html=True)
        # FIX: Changed use_column_width to use_container_width
        st.image("https://placehold.co/600x400/f0f2f6/1a1a1a?text=Select+Algorithm+and+Parameters", use_container_width=True)


st.markdown("---")
st.markdown("Edge Detection App built using Streamlit and OpenCV. Adjust parameters in the sidebar to observe real-time changes!")
