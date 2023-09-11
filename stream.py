import streamlit as st

from TRT_cudart import *
import cv2
import tempfile
from PIL import Image
import numpy as np

st.write("""
### Face Detection
""")
         
# Upload a video file
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

enggine_path = "/home/airi/yolo/Yolov5_Video_Inference/deploy/models/face_detect.engine"
enggine = TRTEngine(enggine_path)


if video_file is not None:
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())
    
    # Open the video file for reading
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = 0
    # Check if the video file is open
    if cap.isOpened():
        st.write("Video Playback:")
        fps = 0
        fpss = []
        prev_time = 0
        curr_time = 0
        fps_out = st.empty()
        image_out = st.empty()
        # Read and display frames from the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            prev_time = time.time()
            frame = cv2.resize(frame, (width, height))
            output_img = run_tensorrt(enggine, image = frame)
            time.sleep(0.0099)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            fps_out.write(f"FPS:{fps}")
            # cv2.putText(output_img, f'FPS:{fps:.3f}'s, 
            #     (10, 30),  # Position of the text
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.75, 
            #     (0, 0, 0),  # Color in BGR format (green in this case)
            #     thickness=2)

        
            # Display the frame in Streamlit
            image_out.image(output_img, channels="BGR", use_column_width=True)
            
            if k % 20 == 0:
                time.sleep(1)
            
            k += 1

        # Release everything after the job is finished
        cap.release()
        # out.release()
        cv2.destroyAllWindows()
    else:
        st.write("Error: Unable to open the video file.")
else:
    st.write("Please upload a video file to display.")



# #### USE YOUR WEB CAMERA
# img_file_buffer = st.camera_input("Take a picture")

# if img_file_buffer is not None:
#     # To read image file buffer as a PIL Image:
#     image_out = st.empty()
#     img = Image.open(img_file_buffer)

#     # To convert PIL Image to numpy array:
#     img_array = np.array(img)

#     # Check the shape of img_array:
#     # Should output shape: (height, width, channels)
#     height, width, channel = img_array.shape
#     frame = cv2.resize(img_array, (width, height))
#     output_img = run_tensorrt(enggine_path = "/home/airi/yolo/Yolov5_Video_Inference/deploy/models/face_convert.onnx", image = frame)
#     image_out.image(output_img, channels="RGB", use_column_width=True)




