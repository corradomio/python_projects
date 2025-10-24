from collections import defaultdict

import cv2
import torch


def save_image(fname: str, fsave: str, target, idx2label: dict):
    gt_im = cv2.imread(fname)
    gt_im_copy = gt_im.copy()

    # Saving images with ground truth boxes
    for idx, box in enumerate(target['bboxes']):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
        cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
        text = idx2label[target['labels'][idx].detach().cpu().item()]
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        text_w, text_h = text_size
        cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
        cv2.putText(gt_im, text=idx2label[target['labels'][idx].detach().cpu().item()],
                    org=(x1 + 5, y1 + 15),
                    thickness=1,
                    fontScale=1,
                    color=[0, 0, 0],
                    fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.putText(gt_im_copy, text=text,
                    org=(x1 + 5, y1 + 15),
                    thickness=1,
                    fontScale=1,
                    color=[0, 0, 0],
                    fontFace=cv2.FONT_HERSHEY_PLAIN)
    cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
    # cv2.imwrite('samples/output_frcnn_gt_{}.png'.format(sample_index), gt_im)
    cv2.imwrite(fsave, gt_im)
# end