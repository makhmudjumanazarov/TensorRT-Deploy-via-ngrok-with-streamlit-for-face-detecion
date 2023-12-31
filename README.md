### Important
  I built a model for Face Detection using YOLOv8. First I exported the YOLOv8 model to ONNX format, then I exported the ONNX format model to TensorRT format. As a result, when i predicted the video through the TensorRT model, the FPS went up to 170 and deployed via ngrok using streamlit.
  - Trained YOLOv8 model for Face Detection: <a href= "https://drive.google.com/file/d/17BhPnTdBkKJH7UF6qD5dSp6E0Ag7dqFg/view?usp=sharing"> face.pt </a>
  - ONNX model: <a href= "https://drive.google.com/file/d/1TR48GOOgYUzMM1fgF_B6ooao5zdE0xpC/view?usp=sharing"> face.onnx </a>
  - TensorRT  model: <a href= "https://drive.google.com/file/d/1bL9nGekteTBkzy0Ocff0E8JZHMhW36vo/view?usp=sharing"> face_detect.engine </a>
  - Code to convert YOLOv8 model to ONNX format:  <a href= "https://github.com/makhmudjumanazarov/TensorRT-Deploy-via-ngrok-with-streamlit-for-face-detecion/blob/main/pytorch_convert_onnx.py"> pytorch_convert_onnx.py </a>
  
  - Code to convert ONNX model TensorRT format: <a href= "https://github.com/makhmudjumanazarov/TensorRT-Deploy-via-ngrok-with-streamlit-for-face-detecion/blob/main/convert_onnx_to_engineresnet.py"> convert_onnx_to_engineresnet.py </a>

### Result
 * **Video inference**: <a href= "https://www.youtube.com/shorts/ZCYN_04dW-w"> Result Video </a>
 * **Link**: https://tidy-caiman-enabling.ngrok-free.app
 * **Video for Test**: https://github.com/makhmudjumanazarov/TensorRT-Deploy-via-ngrok-with-streamlit-for-face-detecion/blob/main/images/GIF.mp4
 
### Steps to Use
<br />
<b>Step 1.</b> Clone <a href= "https://github.com/makhmudjumanazarov/TensorRT-Deploy-via-ngrok-with-streamlit-for-face-detecion.git">this repository </a>
via Terminal
<br/><br/>
<b>Step 2.</b> Create a new virtual environment 
<pre>
python -m venv TensorRT
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source TensorRT/bin/activate # Linux
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install -r requirements.txt 
pip install ipykernel
python -m ipykernel install --user --name=TensorRT
</pre>
<br/>
<b>Step 5.</b> Run streamlit on localhost by running the stream.py file via terminal command (You can select an optional port)
<pre>
streamlit run stream.py --server.port 8520
</pre>

<br/>
<b>Step 6.</b> Open another page in the terminal (it should be the same as the path above). 
<pre>
  - Sign up: https://ngrok.com/
  - Connect your account: 
                        1. ngrok config add-authtoken your token
                        2. ngrok http 8520     
                        
</pre>
<br/>

