**Edge Detection Playground**

**Project Description**

This project is an interactive web application built with Streamlit and the OpenCV library for computer vision experimentation. It allows users to upload an image and apply three classic edge detection algorithms—Canny, Sobel, and Laplacian—while dynamically adjusting their parameters in real-time.

The application uses a clean, professional Light Mode theme for high readability and a simple, effective user experience.

**Key Features**

Three Edge Detection Algorithms: Canny, Sobel, and Laplacian.

Real-time Interactivity: All algorithm parameters (thresholds, kernel size, sigma) can be adjusted using sidebar controls, causing the output image to update instantly.

Side-by-Side Comparison: The Original Input Image and the Processed Edge-Detected Output Image are displayed side-by-side for easy comparison.

Responsive UI: The layout is fully responsive, ensuring optimal viewing on both desktop and mobile devices.

Clean Code: The source code is appropriately commented for clarity and maintainability.

**Setup and Installation Instructions**

This guide assumes you are using Python 3.9+ and have Git installed.

1. Project Setup

Place the edge_detection_app.py file into your main project directory.

2. Set up Virtual Environment

It is crucial to use a virtual environment (venv) to isolate dependencies:

Create the virtual environment
python -m venv venv

Activate the virtual environment (Windows)
.\venv\Scripts\activate


3. Install Dependencies

Install all necessary packages, including Streamlit, OpenCV, NumPy, and Pillow:

.\venv\Scripts\python.exe -m pip install streamlit opencv-python-headless numpy Pillow


How to Run the Application

Ensure the virtual environment is activated (using the command in step 2 above).

Run the application using the Streamlit command, referencing the correct file name:

.\venv\Scripts\python.exe -m streamlit run edge_detection_app.py


The application will open automatically in your default web browser at http://localhost:8501.


Application Sc
