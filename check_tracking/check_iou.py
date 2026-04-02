def iou_xyxy(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    areaB = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = areaA + areaB - inter_area
    return (inter_area / union) if union > 0 else 0.0

print(iou_xyxy(
    [0,0,1,1],
    [0,0,1,1]
))

print(iou_xyxy(
    [0,0,1,1],
    [0.5,0,1.5,1]
))

print(iou_xyxy(
    [0,0,1,1],
    [0.5,0.5,1.5,1.5]
))

print(iou_xyxy(
    [0,0,1,1],
    [-0.5,0.5,0.5,1.5]
))

print(iou_xyxy(
    [0,0,1,1],
    [-0.5,-0.5,0.5,0.5]
))

print(iou_xyxy(
    [0,0,1,1],
    [0.5,-0.5,1.5,0.5]
))

print(iou_xyxy(
    [0,0,1,1],
    [1,0,2,1]
))
