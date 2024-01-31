import numpy as np
import torch
import cv2
import sys

sys.path.append('../core')
from AGRNet import AGRNet
from config import get_config
from demo_utils import norm_kp, computeNN, draw_matching, ExtractSIFT

torch.set_grad_enabled(False)


def demo(opt, img1_path, img2_path, weight):
    print("=======> Loading images")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    print("=======> Generating initial matching")
    SIFT = ExtractSIFT(num_kp=2000)

    kpts1, desc1 = SIFT.run(img1)
    kpts2, desc2 = SIFT.run(img2)

    idx_sort, ratio_test, mutual_nearest = computeNN(desc1, desc2)

    kpts2 = kpts2[idx_sort[1], :]

    cx1 = (img1.shape[1] - 1.0) * 0.5
    cy1 = (img1.shape[0] - 1.0) * 0.5
    f1 = max(img1.shape[1] - 1.0, img1.shape[0] - 1.0)

    cx2 = (img2.shape[1] - 1.0) * 0.5
    cy2 = (img2.shape[0] - 1.0) * 0.5
    f2 = max(img2.shape[1] - 1.0, img2.shape[0] - 1.0)

    kpts1_n = norm_kp(cx1, cy1, f1, f1, kpts1)
    kpts2_n = norm_kp(cx2, cy2, f2, f2, kpts2)

    xs = np.concatenate([kpts1_n, kpts2_n], axis=-1)

    print("=======> Loading pretrained model")
    model = AGRNet(opt).to('cuda')
    state_dict = torch.load(weight, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    print("=======> Pruning")
    data = {}
    data['xs'] = torch.from_numpy(xs[None, None]).float().to('cuda')
    res_logits, _ = model(data)
    print("=======> Done")

    # init matching
    init_matching = draw_matching(img1, img2, kpts1, kpts2)
    cv2.imwrite('./init_matching.png', init_matching)

    # 1st pruning
    mask0 = torch.tanh(res_logits[0][0].cpu()).numpy() > 0
    matching = draw_matching(img1, img2, kpts1[mask0], kpts2[mask0])
    cv2.imwrite('./1st_prune_matching.png', matching)

    # 2nd pruning
    mask1 = torch.tanh(res_logits[1][0].cpu()).numpy() > 0
    matching = draw_matching(img1, img2, kpts1[mask1], kpts2[mask1])
    cv2.imwrite('./2nd_prune_matching.png', matching)

    # picking up inliers from candidates
    mask_final = torch.tanh(res_logits[2][0].cpu()).numpy() > 0
    matching = draw_matching(img1, img2, kpts1[mask_final], kpts2[mask_final])
    cv2.imwrite('./inliers.png', matching)


if __name__ == '__main__':
    opt, unparsed = get_config()

    img1_path = './img1.png'
    img2_path = './img2.png'
    weight = "../model/yfcc/model_best.pth"

    demo(opt, img1_path, img2_path, weight)
