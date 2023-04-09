import torch_directml
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime as ort

# Load decoder model into Intel CPU
onnx_model_path = "mask_decoder.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
# Load SAM model into Intel dGPU
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=torch_directml.device())
predictor = SamPredictor(sam)

# Webcam Loop
#cap = cv2.VideoCapture(0)
split_size = 1
onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
onnx_has_mask_input = np.zeros(1, dtype=np.float32)

while True:
    #ret, frame = cap.read()
    frame = cv2.imread('notebooks/images/dog.jpg')
    H, W, C =frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # get the image_embedding
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    # prepare the prompt for the decoder
    onnx_coord = np.zeros([1,split_size*split_size+1,2]).astype(np.float32)
    onnx_label = (np.zeros([1,split_size*split_size+1])-1).astype(np.float32)
    for i in range(split_size):
        for j in range(split_size):
            input_point = np.array([[int(W/(split_size+2)*(i+1)), int(H/(split_size+2)*(j+1))]])
            input_label = np.array([1])
            onnx_coord[0,i*split_size+j] = input_point.astype(np.float32)
            onnx_label[0,i*split_size+j] = input_label.astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    # Use the mask output from the previous run. It is already in the correct form for input to the ONNX model.
    onnx_mask_input = low_res_logits
    onnx_has_mask_input = np.ones(1, dtype=np.float32)
    masks = masks > predictor.model.mask_threshold
    # Show the masks
    color = np.random.random(3)*255
    h, w = masks.shape[-2:]
    masks_image = masks.reshape(h, w, 1) * color.reshape(1, 1, -1).astype(np.uint8)
    frame = cv2.addWeighted(frame, 1, masks_image, 0.5, 0)
    cv2.imshow("SAM on Intel A770 Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#cap.release()
cv2.destroyAllWindows()

