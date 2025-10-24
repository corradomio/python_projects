import os
import pickle
import random
from pprint import pprint
from random import randrange

import cv2
from tqdm import tqdm
from torchx.nn.cvision.bbox import save_image
from skorchx import NeuralNet
from voc import VOCDataset


def infer():
    # from torchx.nn.cvision.bbox import save_image
    if not os.path.exists('samples'):
        os.mkdir('samples')

    voc_ds = VOCDataset(
        'train',
        im_dir='E:/Datasets/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages',
        ann_dir='E:/Datasets/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations',
    )

    with open('faster-rcnn_pre.pkl', 'rb') as f:
        net: NeuralNet = pickle.load(f)

    net.module_.roi_head.low_score_threshold = 0.7  # 0.4

    for sample_count in tqdm(range(10)):
        random_idx = random.randint(0, len(voc_ds))
        im, target, fname = voc_ds.get_image(random_idx)
        im = im.unsqueeze(0).float()

        save_image(fname, 'samples/output_frcnn_gt_{}.png'.format(sample_count), target, voc_ds.idx2label)

        # gt_im = cv2.imread(fname)
        # gt_im_copy = gt_im.copy()
        #
        # # Saving images with ground truth boxes
        # for idx, box in enumerate(target['bboxes']):
        #     x1, y1, x2, y2 = box.detach().cpu().numpy()
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #
        #     cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
        #     cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
        #     text = voc_ds.idx2label[target['labels'][idx].detach().cpu().item()]
        #     text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        #     text_w, text_h = text_size
        #     cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
        #     cv2.putText(gt_im, text=voc_ds.idx2label[target['labels'][idx].detach().cpu().item()],
        #                 org=(x1 + 5, y1 + 15),
        #                 thickness=1,
        #                 fontScale=1,
        #                 color=[0, 0, 0],
        #                 fontFace=cv2.FONT_HERSHEY_PLAIN)
        #     cv2.putText(gt_im_copy, text=text,
        #                 org=(x1 + 5, y1 + 15),
        #                 thickness=1,
        #                 fontScale=1,
        #                 color=[0, 0, 0],
        #                 fontFace=cv2.FONT_HERSHEY_PLAIN)
        # cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        # cv2.imwrite('samples/output_frcnn_gt_{}.png'.format(sample_count), gt_im)

        # Getting predictions from trained model
        rpn_output, frcnn_output = net.predict(im)
        bboxes = frcnn_output['bboxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']

        save_image(fname, 'samples/output_frcnn_{}.png'.format(sample_count), frcnn_output, voc_ds.idx2label)

        # im = cv2.imread(fname)
        # im_copy = im.copy()
        #
        # # Saving images with predicted boxes
        # for idx, box in enumerate(boxes):
        #     x1, y1, x2, y2 = box.detach().cpu().numpy()
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
        #     cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
        #     text = '{} : {:.2f}'.format(voc_ds.idx2label[labels[idx].detach().cpu().item()],
        #                                scores[idx].detach().cpu().item())
        #     # text = '{} : {:.2f}'.format(voc_ds.idx2label[labels[idx]],
        #     #                            scores[idx])
        #     text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        #     text_w, text_h = text_size
        #     cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
        #     cv2.putText(im, text=text,
        #                 org=(x1 + 5, y1 + 15),
        #                 thickness=1,
        #                 fontScale=1,
        #                 color=[0, 0, 0],
        #                 fontFace=cv2.FONT_HERSHEY_PLAIN)
        #     cv2.putText(im_copy, text=text,
        #                 org=(x1 + 5, y1 + 15),
        #                 thickness=1,
        #                 fontScale=1,
        #                 color=[0, 0, 0],
        #                 fontFace=cv2.FONT_HERSHEY_PLAIN)
        # cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        # cv2.imwrite('samples/output_frcnn_{}.jpg'.format(sample_count), im)


def infer_plain():

    voc_ds = VOCDataset(
        'train',
        im_dir='E:/Datasets/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages',
        ann_dir='E:/Datasets/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations',
    )

    with open('faster-rcnn_pre.pkl', 'rb') as f:
        net: NeuralNet = pickle.load(f)

    net.module_.roi_head.low_score_threshold = 0.4

    n = len(voc_ds)
    im_tensor, targets = voc_ds.get(randrange(n))

    predictions = net.predict(im_tensor)
    pprint(predictions)

    print(type(net))


def main():
    # infer_plain()
    infer()
    pass


if __name__ == "__main__":
    main()

