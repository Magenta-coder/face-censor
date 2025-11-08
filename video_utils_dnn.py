import os, cv2, numpy as np

# =========================
# Utils
# =========================
def _pixelate_roi(frame, x, y, w, h, block=16):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0: return
    down_w = max(1, w // block)
    down_h = max(1, h // block)
    small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    frame[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _rotate_frame(frame, deg):
    if deg == 0:   return frame
    if deg == 90:  return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def _load_dnn(model_dir="models",
              proto="deploy.prototxt",
              weights="res10_300x300_ssd_iter_140000_fp16.caffemodel"):
    p = os.path.join(model_dir, proto)
    w = os.path.join(model_dir, weights)
    if not (os.path.exists(p) and os.path.exists(w)):
        raise FileNotFoundError("Model DNN tidak ditemukan di 'models/'.")
    net = cv2.dnn.readNetFromCaffe(p, w)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def _detect_dnn_boxes(frame, net, conf_thresh=0.5, input_size=(300,300)):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, input_size, (104,177,123), swapRB=False, crop=False)
    net.setInput(blob)
    det = net.forward()
    boxes = []
    for i in range(det.shape[2]):
        conf = float(det[0,0,i,2])
        if conf < conf_thresh: continue
        x1, y1, x2, y2 = (det[0,0,i,3:7] * np.array([w, h, w, h])).astype(int)
        x1 = max(0,x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
        if x2 > x1 and y2 > y1:
            boxes.append([x1,y1,x2,y2,conf])
    return boxes

def _flip_boxes_horiz(boxes, img_w):
    out = []
    for x1,y1,x2,y2,conf in boxes:
        fx1 = img_w - 1 - x2
        fx2 = img_w - 1 - x1
        out.append([fx1,y1,fx2,y2,conf])
    return out

def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    area_b = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter / float(area_a + area_b - inter)

def _nms(boxes, iou_thr=0.45):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if _iou(best, b) < iou_thr]
    return keep

class IOUTracker:
    def __init__(self, iou_thr=0.3, max_miss=5):
        self.iou_thr = iou_thr
        self.max_miss = max_miss
        self.tracks = {}
        self.next_id = 1

    def update(self, dets):
        assigned = set()
        updated = {}
        for tid, t in self.tracks.items():
            best_j, best_iou = -1, 0.0
            for j, d in enumerate(dets):
                if j in assigned: continue
                iou = _iou(t["box"], d)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= self.iou_thr:
                updated[tid] = {"box": dets[best_j][:4], "miss": 0}
                assigned.add(best_j)
            else:
                miss = t["miss"] + 1
                if miss <= self.max_miss:
                    updated[tid] = {"box": t["box"], "miss": miss}
        for j, d in enumerate(dets):
            if j in assigned: continue
            updated[self.next_id] = {"box": d[:4], "miss": 0}
            self.next_id += 1
        self.tracks = updated
        return [v["box"] for v in self.tracks.values()]

# =========================
# Pipeline utama (PIXELATE ONLY) + progress callback
# =========================
def censor_video_dnn(
    input_path, output_path,
    block_size=18,
    pad_ratio=0.25,
    rotate=0,
    conf=0.5,
    nms_iou=0.45,
    use_flip=True,
    track_iou=0.3,
    track_max_miss=5,
    progress_fn=None   # <— callback: progress_fn(done_frames, total_frames)
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input tidak ditemukan: {input_path}")

    net = _load_dnn()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Gagal membuka video input.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        import math
        if math.isnan(fps) or fps == 0: fps = 25.0
    except Exception:
        if fps == 0: fps = 25.0

    ok, frame0 = cap.read()
    if not ok: raise RuntimeError("Video kosong.")
    frame0 = _rotate_frame(frame0, rotate)
    H, W = frame0.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    if not out.isOpened(): raise RuntimeError("Gagal membuat video output.")

    tracker = IOUTracker(iou_thr=track_iou, max_miss=track_max_miss)

    def detect_ensemble(f):
        boxes = _detect_dnn_boxes(f, net, conf_thresh=conf)
        if use_flip:
            f_flip = cv2.flip(f, 1)
            boxes_flip = _detect_dnn_boxes(f_flip, net, conf_thresh=conf)
            boxes_flip = _flip_boxes_horiz(boxes_flip, f.shape[1])
            boxes.extend(boxes_flip)
        return _nms(boxes, iou_thr=nms_iou)

    def apply_pixelate(f, boxes):
        for x1,y1,x2,y2 in boxes:
            bw, bh = (x2-x1+1), (y2-y1+1)
            pad = int(max(bw, bh) * pad_ratio)
            px1 = max(0, x1 - pad); py1 = max(0, y1 - pad)
            px2 = min(W-1, x2 + pad); py2 = min(H-1, y2 + pad)
            _pixelate_roi(f, px1, py1, px2-px1+1, py2-py1+1, block=max(6, int(block_size)))

    # frame ke-0
    det0 = detect_ensemble(frame0)
    boxes_t0 = tracker.update(det0)
    apply_pixelate(frame0, boxes_t0)
    out.write(frame0)
    done = 1
    if progress_fn: progress_fn(done, total_frames)

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = _rotate_frame(frame, rotate)
        dets = detect_ensemble(frame)
        boxes = tracker.update(dets)
        apply_pixelate(frame, boxes)
        out.write(frame)
        done += 1
        if progress_fn and (done % 3 == 0):
            progress_fn(done, total_frames)

    cap.release(); out.release()
    if progress_fn: progress_fn(total_frames if total_frames else done, total_frames)
    return output_path