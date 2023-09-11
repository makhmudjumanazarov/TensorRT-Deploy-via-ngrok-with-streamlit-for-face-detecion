import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart
from numpy import ndarray
from matplotlib import pyplot as plt
import time

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

@dataclass
class Tensor:
    name: str
    dtype: np.dtype
    shape: Tuple
    cpu: ndarray
    gpu: int
class TRTEngine:
    def __init__(self, weight: Union[str, Path]) -> None:
        self.weight = Path(weight) if isinstance(weight, str) else weight
        status, self.stream = cudart.cudaStreamCreate()
        assert status.value == 0
        self.__init_engine()
        self.__init_bindings()
        self.__warm_up()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())
        context = model.create_execution_context()

        names = [model.get_binding_name(i) for i in range(model.num_bindings)]
        self.num_bindings = model.num_bindings
        self.bindings: List[int] = [0] * self.num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(model.num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]

    def __init_bindings(self) -> None:
        dynamic = False
        inp_info = []
        out_info = []
        out_ptrs = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                dynamic |= True
            if not dynamic:
                cpu = np.empty(shape, dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            else:
                cpu, gpu = np.empty(0), 0
            inp_info.append(Tensor(name, dtype, shape, cpu, gpu))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if not dynamic:
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
                out_ptrs.append(gpu)
            else:
                cpu, gpu = np.empty(0), 0
            out_info.append(Tensor(name, dtype, shape, cpu, gpu))

        self.is_dynamic = dynamic
        self.inp_info = inp_info
        self.out_info = out_info
        self.out_ptrs = out_ptrs

    def __warm_up(self) -> None:
        if self.is_dynamic:
            print('You engine has dynamic axes, please warm up by yourself !')
            return
        for _ in range(10):
            inputs = []
            for i in self.inp_info:
                inputs.append(i.cpu)
            self.__call__(inputs)

    def set_profiler(self, profiler: Optional[trt.IProfiler]) -> None:
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def __call__(self, *inputs) -> Union[Tuple, ndarray]:
        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[ndarray] = [
            np.ascontiguousarray(i) for i in inputs
        ]

        for i in range(self.num_inputs):
            if self.is_dynamic:
                self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))
                status, self.inp_info[i].gpu = cudart.cudaMallocAsync(contiguous_inputs[i].nbytes, self.stream)
                assert status.value == 0
            cudart.cudaMemcpyAsync(
                self.inp_info[i].gpu, contiguous_inputs[i].ctypes.data,
                contiguous_inputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            self.bindings[i] = self.inp_info[i].gpu

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                dtype = self.out_info[i].dtype
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            else:
                cpu = self.out_info[i].cpu
                gpu = self.out_info[i].gpu
            self.bindings[j] = gpu

        # self.context.execute_v2(batch_size=self.num_inputs, bindings=self.bindings)
        self.context.execute_v2(bindings=self.bindings)
        cudart.cudaStreamSynchronize(self.stream)

        for i, o in enumerate(self.out_ptrs):
            cudart.cudaMemcpyAsync(
                self.out_info[i].cpu.ctypes.data, o, self.out_info[i].cpu.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        data_output = tuple([self.out_info[i].cpu for i in range(self.num_outputs)]) if self.num_outputs > 1 else self.out_info[0].cpu
        num_dets, bboxes, scores, labels = (i[0] for i in data_output)
        nums = num_dets.item()
        bboxes = bboxes[:nums]
        scores = scores[:nums]
        labels = labels[:nums]
        return bboxes, scores, labels

def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (0, 0, 0)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)

def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def run_tensorrt(enggine, image):


    H, W = enggine.inp_info[0].shape[-2:]

    # image = cv2.imread(image_path)
    bgr, ratio, dwdh = letterbox(image, (W, H))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = blob(rgb, return_seg=False)
    dwdh = np.array(dwdh * 2, dtype=np.float32)
    tensor = np.ascontiguousarray(tensor)

    # Detection
    results = enggine(tensor)
    bboxes, scores, labels = results
    bboxes -= dwdh
    bboxes /= ratio    

    CLASSES = ('person')

    for (bbox, score, label) in zip(bboxes, scores, labels):
        bbox = bbox.round().astype(np.int32).tolist()
        cls_id = int(label)
        # cls = CLASSES[cls_id]
        color = (0,255,0)
        cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        cv2.putText(image,
                f'Face:{score:.3f}', 
                (bbox[0], bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, 
                [225, 255, 255],
                thickness=2)
    return image

# cap = cv2.VideoCapture("/home/airi/yolo/Yolov5_Video_Inference/supermodel.mp4")
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Initialize the VideoWriter
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

# fps = 0
# fpss = []
# prev_time = 0
# curr_time = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.resize(frame, (width, height))
    
#     prev_time = time.time()
#     output_img = run_tensorrt(enggine_path = "/home/airi/yolo/Yolov5_Video_Inference/deploy/models/face_detect.engine", image = frame)
#     curr_time = time.time()
#     #Write the frame into the file 'output.mp4'
#     out.write(output_img)

#     cv2.putText(output_img, f'FPS:{fps:.3f}', 
#                 (10, 30),  # Position of the text
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.75, 
#                 (0, 255, 0),  # Color in BGR format (green in this case)
#                 thickness=2)
#     cv2.imwrite(f"/home/airi/yolo/Yolov5_Video_Inference/deploy/images/next.png", output_img)
    
#     fps = (1 / (curr_time - prev_time))
#     print("FPS: --", fps)
#     fpss.append(fps)
# plt.plot(fpss)
# plt.savefig('MyPlot.png')

# # Release everything after the job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()
