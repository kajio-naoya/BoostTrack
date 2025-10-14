import os
import sys
import time

import cv2
import numpy as np
import torch

import utils
from default_settings import GeneralSettings, BoostTrackPlusPlusSettings, BoostTrackSettings, get_detector_path_and_im_size
from external.adaptors import detector
from tracker.boost_track import BoostTrack
from types import SimpleNamespace
from dataset import preproc

def track_my_file(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Configure settings (use MOT17 defaults for general videos)
    GeneralSettings.values['dataset'] = 'mot17'
    GeneralSettings.values['use_embedding'] = True
    GeneralSettings.values['use_ecc'] = True
    GeneralSettings.values['test_dataset'] = False

    # Keep BoostTrack++ defaults
    BoostTrackSettings.values['s_sim_corr'] = False
    BoostTrackPlusPlusSettings.values['use_rich_s'] = True
    BoostTrackPlusPlusSettings.values['use_sb'] = True
    BoostTrackPlusPlusSettings.values['use_vt'] = True

    # Build detector
    args = SimpleNamespace(dataset='mot17', test_dataset=False)
    det_path, size = get_detector_path_and_im_size(args)
    det = detector.Detector("yolox", det_path, args.dataset)

    # Device placement
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if det.model is not None:
        det.model.to(device)
        det.model.eval()

    # Frame iterator for either a video file or a directory of images
    is_dir = os.path.isdir(video_path)
    cap = None
    frame_paths = None
    fps = 25.0
    width = None
    height = None
    video_name = os.path.splitext(os.path.basename(video_path.rstrip(os.sep)))[0]

    if is_dir:
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        frame_paths = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.lower().endswith(exts)]
        frame_paths.sort()
        if len(frame_paths) == 0:
            raise RuntimeError("No image frames found in directory.")
        first_img = cv2.imread(frame_paths[0])
        if first_img is None:
            raise RuntimeError("Failed to read the first image frame.")
        height, width = first_img.shape[:2]
        # default fps when reading image sequences
        fps = 25.0
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare overlay video writer (always 1920x1080)
    overlay_path = os.path.join(output_dir, f"{video_name}_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_width, output_height = 1920, 1080
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (output_width, output_height))

    # Initialize tracker
    tracker = BoostTrack(video_name=video_name)
    results = []

    # Preproc config (use same normalization as in dataset loader)
    rgb_means = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Tiling parameters
    OVERLAP_RATIO = 0.1  # 10% overlap
    NMS_IOU_THRESHOLD = 0.7  # IoU threshold for NMS
    NMS_SCORE_THRESHOLD = 0.1  # Minimum confidence threshold

    frame_idx = 0
    total_time = 0.0

    def process_frame(np_img_bgr, frame_id):
        nonlocal total_time
        tag = f"{video_name}:{frame_id}"
        
        # Check if image is large enough to warrant tiling (e.g., > 2000px in any dimension)
        use_tiling = max(height, width) > 2000
        
        if use_tiling:
            # Split into 4x4 tiles with overlap for detection
            grid_n = 4
            tile_h_base = height // grid_n
            tile_w_base = width // grid_n
            overlap_h = int(tile_h_base * OVERLAP_RATIO)
            overlap_w = int(tile_w_base * OVERLAP_RATIO)
            all_detections = []

            # Prepare full-frame tensor once and keep its scale (r_full)
            proc_img_full, r_full = preproc(np_img_bgr, size, rgb_means, std, swap=(2, 0, 1))
            img_tensor = torch.from_numpy(proc_img_full).unsqueeze(0).to(device)

            # Build batch of tiles with overlap
            batch_tiles = []
            batch_rs = []
            batch_offsets = []
            batch_tags = []
            for i in range(grid_n):
                for j in range(grid_n):
                    # Calculate tile boundaries with overlap
                    y1 = max(0, i * tile_h_base - overlap_h)
                    y2 = min(height, (i + 1) * tile_h_base + overlap_h)
                    x1 = max(0, j * tile_w_base - overlap_w)
                    x2 = min(width, (j + 1) * tile_w_base + overlap_w)
                    
                    tile = np_img_bgr[y1:y2, x1:x2]
                    if tile.size == 0:
                        continue
                    proc_tile, r_tile = preproc(tile, size, rgb_means, std, swap=(2, 0, 1))
                    batch_tiles.append(proc_tile)
                    batch_rs.append(r_tile)
                    batch_offsets.append((x1, y1))
                    batch_tags.append(f"{tag}_tile_{i}_{j}")

            with torch.no_grad():
                start = time.time()
                if len(batch_tiles) > 0:
                    batch_tensor = torch.from_numpy(np.stack(batch_tiles)).to(device)
                    batch_preds = det(batch_tensor, batch_tags)
                    
                    # GPU上で座標変換（並列処理）
                    valid_preds = []
                    valid_offsets = []
                    valid_rs = []
                    
                    for i, (pred, (x1, y1), r_tile) in enumerate(zip(batch_preds, batch_offsets, batch_rs)):
                        if pred is not None:
                            valid_preds.append(pred)
                            valid_offsets.append(torch.tensor([x1, y1, x1, y1], device=device, dtype=pred.dtype))
                            valid_rs.append(r_tile)
                    
                    if valid_preds:
                        # GPU上で一括座標変換
                        stacked_preds = torch.cat(valid_preds, dim=0)
                        
                        # 各予測のスケールとオフセットを準備
                        scales = []
                        offsets = []
                        for i, (pred, offset, r_tile) in enumerate(zip(valid_preds, valid_offsets, valid_rs)):
                            scales.extend([r_tile] * len(pred))
                            # 各検出に対してオフセットを繰り返し
                            for _ in range(len(pred)):
                                offsets.append(offset)
                        
                        stacked_scales = torch.tensor(scales, device=device, dtype=stacked_preds.dtype).unsqueeze(1)
                        stacked_offsets = torch.stack(offsets, dim=0)
                        
                        # 座標変換をGPU上で並列実行
                        stacked_preds[:, :4] /= stacked_scales
                        stacked_preds[:, :4] += stacked_offsets
                        
                        # 一度だけCPU転送
                        combined_pred = stacked_preds.cpu().numpy()
                    else:
                        combined_pred = None
                
                # Apply NMS to remove overlapping detections
                if combined_pred is not None and len(combined_pred) > 1:
                    # Convert to format for cv2.dnn.NMSBoxes: [x, y, w, h]
                    boxes = combined_pred[:, :4].copy()
                    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # w = x2 - x1
                    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # h = y2 - y1
                    
                    scores = combined_pred[:, 4]
                    
                    # Apply NMS with IoU threshold
                    indices = cv2.dnn.NMSBoxes(
                        boxes.tolist(), 
                        scores.tolist(), 
                        score_threshold=NMS_SCORE_THRESHOLD,
                        nms_threshold=NMS_IOU_THRESHOLD
                    )
                    
                    if len(indices) > 0:
                        combined_pred = combined_pred[indices.flatten()]
                    else:
                        combined_pred = None
                
                # Filter out detections that are outside image bounds
                if combined_pred is not None:
                    valid_mask = (
                        (combined_pred[:, 0] >= 0) & (combined_pred[:, 1] >= 0) &
                        (combined_pred[:, 2] < width) & (combined_pred[:, 3] < height) &
                        (combined_pred[:, 2] > combined_pred[:, 0]) & (combined_pred[:, 3] > combined_pred[:, 1])
                    )
                    if valid_mask.any():
                        combined_pred = combined_pred[valid_mask]
                        # Clip to image bounds to avoid float drifts
                        combined_pred[:, 0] = np.clip(combined_pred[:, 0], 0, width - 1)
                        combined_pred[:, 1] = np.clip(combined_pred[:, 1], 0, height - 1)
                        combined_pred[:, 2] = np.clip(combined_pred[:, 2], 0, width - 1)
                        combined_pred[:, 3] = np.clip(combined_pred[:, 3], 0, height - 1)
                        # Map to full-frame preprocessed coordinates (so tracker.update rescales correctly)
                        combined_pred[:, :4] *= r_full
                    else:
                        combined_pred = None

                targets = tracker.update(combined_pred, img_tensor, np_img_bgr, tag)
                tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])
                total_time += time.time() - start
        else:
            # Original single-image processing
            proc_img, _ = preproc(np_img_bgr, size, rgb_means, std, swap=(2, 0, 1))
            img_tensor = torch.from_numpy(proc_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                start = time.time()
                pred = det(img_tensor, tag)
                if pred is None:
                    # write empty overlay and continue
                    final_canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    writer.write(final_canvas)
                    return

                targets = tracker.update(pred, img_tensor, np_img_bgr, tag)
                tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])
                total_time += time.time() - start

        # Draw overlay with ID-based colors (no text)
        canvas = np_img_bgr.copy()
        for (x, y, w, h), tid, sc in zip(tlwhs, ids, confs):
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            
            # Generate color based on ID using HSV color space
            hue = (int(tid) * 137) % 180  # Use golden ratio for good distribution
            color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(map(int, color_bgr))
            
            cv2.rectangle(canvas, p1, p2, color, 3)
        
        # Resize to 1920x1080 while maintaining aspect ratio
        scale = min(output_width / width, output_height / height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create 1920x1080 canvas with black padding
        final_canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        y_offset = (output_height - new_h) // 2
        x_offset = (output_width - new_w) // 2
        final_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        writer.write(final_canvas)

        # Save results entry
        results.append((frame_id, tlwhs, ids, confs))

    if is_dir:
        for fp in frame_paths:
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx}")
            frame = cv2.imread(fp)
            if frame is None:
                continue
            frame_idx += 1
            process_frame(frame, frame_idx)
    else:
        while True:
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx}")
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            process_frame(frame, frame_idx)

    # Cleanup IO
    if cap is not None:
        cap.release()
    writer.release()

    # Persist caches
    det.dump_cache()
    tracker.dump_cache()

    # Save MOT-format and CSV results
    mot_path = os.path.join(output_dir, f"{video_name}.txt")
    utils.write_results_no_score(mot_path, results)

    csv_path = os.path.join(output_dir, f"{video_name}.csv")
    # Prepare flattened, sorted entries (frame asc, id asc) and write CSV with integer ids
    flat = []
    for frame_id, tlwhs, ids, confs in results:
        for (x, y, w, h), tid, sc in zip(tlwhs, ids, confs):
            flat.append(
                (
                    int(frame_id),
                    int(tid),
                    round(x, 1),
                    round(y, 1),
                    round(w, 1),
                    round(h, 1),
                    round(sc, 2),
                )
            )
    flat.sort(key=lambda r: (r[0], r[1]))
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("frame,id,x,y,w,h,score\n")
        for frame, tid, x, y, w, h, sc in flat:
            f.write(f"{frame},{tid},{x},{y},{w},{h},{sc}\n")

    print(f"Finished. Results: {mot_path}, {csv_path}; Overlay video: {overlay_path}")

if __name__ == "__main__":
    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    track_my_file(video_path, output_dir)