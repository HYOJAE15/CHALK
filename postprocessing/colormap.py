import os
import cv2
import numpy as np

# 1) 스크립트 위치 기준 베이스 디렉터리
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2) 입력/출력 폴더 정의
left_folder = os.path.join(base_dir, 'leftImg8bit')
gt_folder   = os.path.join(base_dir, 'gtFine')
out_left    = os.path.join(base_dir, 'colormap_leftImg8bit')
out_gt      = os.path.join(base_dir, 'colormap_gtFine')

# 3) 출력 폴더 생성
for d in (out_left, out_gt):
    os.makedirs(d, exist_ok=True)

# 4) 투명도와 클래스별 컬러 맵 (BGR)
alpha = 0.4
color_map = {
    0: (0,   0,   0  ),  # background (투명)
    1: (0,   0, 255),   # crack          → 빨강
    2: (0, 255,   0),   # efflorescence  → 초록
    3: (0, 255, 255),   # rebar-exposure → 노랑
    4: (255, 0,   0),   # spalling       → 파랑
}

# 5) 파일 처리 루프
for fname in os.listdir(left_folder):
    if not fname.endswith('_leftImg8bit.png'):
        continue

    base = fname.replace('_leftImg8bit.png', '')
    img_path = os.path.join(left_folder, fname)
    gt_path  = os.path.join(gt_folder,   base + '_gtFine_labelIds.png')
    if not os.path.isfile(gt_path):
        print(f"Missing GT: {gt_path}, skipping.")
        continue

    # 6) 이미지 로드
    img = cv2.imread(img_path)
    gt  = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    h, w = gt.shape

    # 7) leftImg8bit 위에 반투명 컬러맵 오버레이
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in color_map.items():
        if cls_id == 0:
            continue
        mask[gt == cls_id] = color
    overlay = cv2.addWeighted(img, 1 - alpha, mask, alpha, 0)

    out_overlay = os.path.join(out_left, base + '_leftImg8bit_colormap.png')
    cv2.imwrite(out_overlay, overlay)
    print(f"Saved overlay: {out_overlay}")

    # 8) gtFine 순수 컬러맵 이미지 생성
    colormap = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in color_map.items():
        colormap[gt == cls_id] = color
    out_cm = os.path.join(out_gt, base + '_gtFine_colormap.png')
    cv2.imwrite(out_cm, colormap)
    print(f"Saved colormap: {out_cm}")

print("All processing complete.")
