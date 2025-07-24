import os
import cv2
import numpy as np

# 폴더 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
gt_folder   = os.path.join(base_dir, 'gtFine')
pred_folder = os.path.join(base_dir, 'predicted')  # 예측 결과(labelIds와 동일한 형태)

# IoU 저장 리스트
iou_list = []

# IoU 계산
for fname in os.listdir(gt_folder):
    if not fname.endswith('_gtFine_labelIds.png'):
        continue

    base = fname.replace('_gtFine_labelIds.png', '')
    gt_path = os.path.join(gt_folder, fname)
    pred_path = os.path.join(pred_folder, base + '_pred_labelIds.png')

    if not os.path.isfile(pred_path):
        print(f"Missing prediction: {pred_path}, skipping.")
        continue

    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)

    # 클래스 0 (배경) 제외
    valid_mask = (gt > 0) | (pred > 0)

    intersection = np.logical_and(gt == pred, valid_mask).sum()
    union = valid_mask.sum()

    if union == 0:
        iou = np.nan
        print(f"{fname}: IoU 계산 불가 (클래스 없음)")
    else:
        iou = intersection / union
        print(f"{fname}: IoU = {iou:.4f}")
        iou_list.append(iou)

# 평균 IoU 출력
valid_ious = [i for i in iou_list if not np.isnan(i)]
if valid_ious:
    mean_iou = np.mean(valid_ious)
    print(f"\n전체 평균 IoU: {mean_iou:.4f}")
else:
    print("\n평균 IoU를 계산할 수 없습니다 (유효한 비교 없음).")
