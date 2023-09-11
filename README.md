### TensorRT-Deploy-via-ngrok-with-streamlit-for-face-detecion
Description:
  I built a model for Face Detection using YOLOv8. Then i exported YOLOv8 model to engine format.
  - Trained YOLOv8 model for Face Detection: <a href= "https://drive.google.com/file/d/17BhPnTdBkKJH7UF6qD5dSp6E0Ag7dqFg/view?usp=sharing"> face.pt </a>
  - Code to convert YOLOv8 model to ONNX format:  <a href= "https://github.com/makhmudjumanazarov/TensorRT-Deploy-via-ngrok-with-streamlit-for-face-detecion/blob/main/pytorch_convert_onnx.py"> pytorch_convert_onnx.py </a>
  - Code to convert ONNX model TensorRT format: <a href= "https://github.com/makhmudjumanazarov/TensorRT-Deploy-via-ngrok-with-streamlit-for-face-detecion/blob/main/convert_onnx_to_engineresnet.py"> convert_onnx_to_engineresnet.py </a>

#### Steps to Use
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


https://drive.google.com/drive/folders/1V9RF_1FDGvdrt_moMkeoPCzkDj6Ln2oo?usp=sharing
