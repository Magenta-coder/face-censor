import os
import cv2
import numpy as np

def _pixelate_roi(frame, x, y, w, h, block=15):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0: 
        return
    small = cv2.resize(roi, (max(1, w//block), max(1, h//block)), interpolation=cv2.INTER_LINEAR)
    frame[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _blur_roi(frame, x, y, w, h, k=31):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0: 
        return
    k = k if k % 2 == 1 else k + 1
    frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)

def _rotate_frame(frame, deg):
    if deg == 0:   return frame
    if deg == 90:  return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def _load_dnn(model_dir="models", proto="deploy.prototxt", weights="res10_300x300_ssd_iter_140000_fp16.caffemodel"):
    proto_path   = os.path.join(model_dir, proto)
    weights_path = os.path.join(model_dir, weights)
    if not os.path.exists(proto_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            "Model DNN tidak ditemukan. Pastikan 'models/deploy.prototxt' dan "
            "'models/res10_300x300_ssd_iter_140000_fp16.caffemodel' tersedia."
        )
    net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def _detect_faces_dnn(frame, net, conf_thresh=0.5, input_size=(300, 300)):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, input_size, (104.0, 177.0, 123.0),
                                 swapRB=False, crop=False)
    net.setInput(blob)
    det = net.forward()
    boxes = []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        (x1, y1, x2, y2) = (det[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2-x1, y2-y1, conf))
    return boxes

def censor_video_dnn(input_path, output_path, mode="blur", strength=31, padding=20,
                     rotate=0, conf=0.5):
    net = _load_dnn()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Gagal membuka video input.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25.0

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Video kosong.")

    frame0 = _rotate_frame(frame0, rotate)
    H, W = frame0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    if not out.isOpened():
        raise RuntimeError("Gagal membuat video output.")

    def process(f):
        boxes = _detect_faces_dnn(f, net, conf_thresh=conf)
        for (x, y, w, h, c) in boxes:
            x1 = max(0, x - padding); y1 = max(0, y - padding)
            x2 = min(W, x + w + padding); y2 = min(H, y + h + padding)
            if mode == "pixelate":
                _pixelate_roi(f, x1, y1, x2-x1, y2-y1, block=max(4, int(strength)))
            else:
                _blur_roi(f, x1, y1, x2-x1, y2-y1, k=max(3, int(strength) | 1))
        return f

    out.write(process(frame0))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _rotate_frame(frame, rotate)
        out.write(process(frame))

    cap.release()
    out.release()
    return output_path
