import numpy as np
import torch
import cv2
import sys
import os

sys.path.append('../core')
from AGRNet import AGRNet
from config import get_config
from evaluation import eval_nondecompose
from demo_utils import norm_kp, computeNN, draw_matching, draw_logits, ExtractSIFT, get_gt

torch.set_grad_enabled(False)


def demo(model, img1_path, img2_path, K1, K2, T_1to2, flip=False, only_pose=False,
         intermediate=True, figure_logits=True):
    print("=======> Loading images")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    print("=======> Generating initial matching")
    SIFT = ExtractSIFT(num_kp=2000)

    kpts1, desc1 = SIFT.run(img1)
    kpts2, desc2 = SIFT.run(img2)

    idx_sort, _, _ = computeNN(desc1, desc2)

    kpts2 = kpts2[idx_sort[1], :]

    cx1, cy1, fx1, fy1 = K1[0, 2], K1[1, 2], K1[0, 0], K1[1, 1]
    cx2, cy2, fx2, fy2 = K2[0, 2], K2[1, 2], K2[0, 0], K2[1, 1]

    kpts1_n = norm_kp(cx1, cy1, fx1, fy1, kpts1)
    kpts2_n = norm_kp(cx2, cy2, fx2, fy2, kpts2)

    xs = np.concatenate([kpts1_n, kpts2_n], axis=-1)

    print("=======> Pruning")
    data = {}
    data['xs'] = torch.from_numpy(xs[None, None]).float().to('cuda')
    res_logits, res_e_hat = model(data)
    print("=======> Done")

    # get ground truth mask
    residual = get_gt(kpts1_n, kpts2_n, T_1to2)
    mask = residual < 1e-4

    # init image pair
    init_image_pair = draw_matching(img1, img2, np.array([]), np.array([]), flip=flip)
    cv2.imwrite(f'{save_path}_init_image_pair.png', init_image_pair)

    # init matching
    init_matching = draw_matching(img1, img2, kpts1, kpts2, mask, flip=flip)
    cv2.imwrite(f'{save_path}_init_matching.png', init_matching)

    # 1st pruning
    mask0 = torch.tanh(res_logits[0][0].cpu()).numpy() > 0
    dR, dt = T_1to2[:3, :3], T_1to2[:3, 3]
    ret = eval_nondecompose(kpts1_n, kpts2_n, res_e_hat[0][0].cpu().numpy(), dR, dt, res_logits[0][0].cpu().numpy())
    R_err, t_err = ret[0] * 180 / np.pi, ret[1] * 180 / np.pi
    TP = np.sum(mask0.astype(float) * mask.astype(float))
    P = np.sum(mask0.astype(float))
    TP_FN = np.sum(mask.astype(float))
    Precision = TP / P
    Recall = TP / TP_FN
    text = [
        f"AGRNet",
        f"#Precision: {Precision:.3f} [{int(TP)}/{int(P)}]",
        f"#Recall: {Recall:.3f} [{int(TP)}/{int(TP_FN)}]",
        f"#R_err: {R_err:.2f}",
        f"#t_err: {t_err:.2f}",
    ]
    if only_pose:
        text = [
            f"Coarse-level AGR-Block",
            f"#R_err: {R_err:.2f}",
            f"#t_err: {t_err:.2f}",
        ]
    if intermediate:
        matching = draw_matching(img1, img2, kpts1[mask0], kpts2[mask0], mask[mask0], text, flip=flip)
        cv2.imwrite(f'{save_path}_1st_prune_matching.png', matching)
        if figure_logits:
            draw_logits(res_logits[0][0].cpu().numpy(), residual, Precision, Recall, f'{save_path}_1st_prune_logits.png')

    # 2nd pruning
    mask1 = torch.tanh(res_logits[1][0].cpu()).numpy() > 0
    ret = eval_nondecompose(kpts1_n, kpts2_n, res_e_hat[1][0].cpu().numpy(), dR, dt, res_logits[1][0].cpu().numpy())
    R_err, t_err = ret[0] * 180 / np.pi, ret[1] * 180 / np.pi
    TP = np.sum(mask1.astype(float) * mask.astype(float))
    P = np.sum(mask1.astype(float))
    TP_FN = np.sum(mask.astype(float))
    Precision = TP / P
    Recall = TP / TP_FN
    text = [
        f"AGRNet",
        f"#Precision: {Precision:.3f} [{int(TP)}/{int(P)}]",
        f"#Recall: {Recall:.3f} [{int(TP)}/{int(TP_FN)}]",
        f"#R_err: {R_err:.2f}",
        f"#t_err: {t_err:.2f}",
    ]
    if only_pose:
        text = [
            f"Fine-level AGR-Block",
            f"#R_err: {R_err:.2f}",
            f"#t_err: {t_err:.2f}",
        ]
    if intermediate:
        matching = draw_matching(img1, img2, kpts1[mask1], kpts2[mask1], mask[mask1], text, flip=flip)
        cv2.imwrite(f'{save_path}_2nd_prune_matching.png', matching)
        if figure_logits:
            draw_logits(res_logits[1][0].cpu().numpy(), residual, Precision, Recall, f'{save_path}_2nd_prune_logits.png')

    # picking up inliers from candidates
    mask_final = torch.tanh(res_logits[2][0].cpu()).numpy() > 0
    ret = eval_nondecompose(kpts1_n, kpts2_n, res_e_hat[2][0].cpu().numpy(), dR, dt, res_logits[2][0].cpu().numpy())
    R_err, t_err = ret[0] * 180 / np.pi, ret[1] * 180 / np.pi
    TP = np.sum(mask_final.astype(float) * mask.astype(float))
    P = np.sum(mask_final.astype(float))
    TP_FN = np.sum(mask.astype(float))
    Precision = TP / P
    Recall = TP / TP_FN
    text = [
        f"AGRNet",
        f"#Precision: {Precision:.3f} [{int(TP)}/{int(P)}]",
        f"#Recall: {Recall:.3f} [{int(TP)}/{int(TP_FN)}]",
        f"#R_err: {R_err:.2f}",
        f"#t_err: {t_err:.2f}"
    ]
    if only_pose:
        text = [
            f"Joint Predictor",
            f"#R_err: {R_err:.2f}",
            f"#t_err: {t_err:.2f}",
        ]
    matching = draw_matching(img1, img2, kpts1[mask_final], kpts2[mask_final], mask[mask_final], text, flip=flip)
    cv2.imwrite(f'{save_path}_inliers.png', matching)
    if figure_logits:
        draw_logits(res_logits[2][0].cpu().numpy(), residual, Precision, Recall, f'{save_path}_inliers_logits.png')


if __name__ == '__main__':
    opt, unparsed = get_config()

    # specify the index of pair to visualize
    idxs = [200]

    # read pair infos
    pair_file = "yfcc100m_test_buckingham_palace.txt"
    with open(pair_file, 'r') as f:
        pair_infos = [l.split() for l in f.readlines()]

    # specify saving directory
    save_dir = pair_file.split('.')[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize model
    weight = "../model/yfcc/model_best.pth"
    print("=======> Loading pretrained model")
    model = AGRNet(opt).to('cuda')
    state_dict = torch.load(weight, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # extract one by one
    for idx in idxs:
        pair = pair_infos[idx]
        img1_path, img2_path = pair[:2]
        K1 = np.array(pair[2:11]).astype(float).reshape(3, 3)
        K2 = np.array(pair[11:20]).astype(float).reshape(3, 3)
        T_1to2 = np.array(pair[20:]).astype(float).reshape(4, 4)
        save_path = save_dir + f'/AGRNet_{idx}'
        demo(model, img1_path, img2_path, K1, K2, T_1to2, flip=False, only_pose=False, 
             intermediate=False, figure_logits=False)
