import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = torch.from_numpy(desc_ii), torch.from_numpy(desc_jj)
    d1 = (desc_ii ** 2).sum(1)
    d2 = (desc_jj ** 2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2 * torch.matmul(desc_ii, desc_jj.transpose(0, 1))).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:, 0]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)
    nnIdx2 = nnIdx2.squeeze()
    mutual_nearest = (nnIdx2[nnIdx1] == torch.arange(0, nnIdx1.shape[0]).long()).numpy()
    ratio_test = (distVals[:, 0] / distVals[:, 1].clamp(min=1e-10)).numpy()
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.numpy()]
    return idx_sort, ratio_test, mutual_nearest


def draw_matching(img1, img2, pt1, pt2, mask=None, text=None, flip=False):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if flip:
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1 + w2] = img2
    else:
        vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
        vis[:h1, :w1] = img1
        vis[h1:h1 + h2, :w2] = img2

    green = (0, 255, 0)
    red = (0, 0, 255)
    thickness = 1

    for i in range(pt1.shape[0]):
        x1 = int(pt1[i, 0])
        y1 = int(pt1[i, 1])
        if flip:
            x2 = int(pt2[i, 0] + w1)
            y2 = int(pt2[i, 1])
        else:
            x2 = int(pt2[i, 0])
            y2 = int(pt2[i, 1] + h1)
        color = green
        if mask is not None:
            color = green if mask[i] else red
        cv2.line(vis, (x1, y1), (x2, y2), color, int(thickness))

    if text is not None:
        for i in range(len(text)):
            cv2.putText(vis, text[i], (3, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return vis


def draw_logits(logits, residual, precision=None, recall=None, save_path=None):
    # 对 residual 进行升序排列，得到排序的索引
    sorted_indices = np.argsort(residual)

    # 根据 residual 排序的索引，对 logits 和 residual 进行重新排序
    sorted_logits = logits[sorted_indices]
    sorted_residual = residual[sorted_indices]

    # 判断 inlier 和 outlier
    inlier_mask = sorted_residual < 1e-4
    outlier_mask = ~inlier_mask

    # 绘制柱状图
    fig = plt.bar(range(len(logits)), sorted_logits, color=np.where(inlier_mask, 'green', 'red'))

    # 计算 inlier 和 outlier 的数量
    inlier_count = np.sum(inlier_mask)
    outlier_count = np.sum(outlier_mask)

    # 标注 inlier 和 outlier 的数量，精度和召回率
    plt.text(0.85, 0.95, f'Inliers: {inlier_count}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, color='green', fontsize=8)
    plt.text(0.85, 0.9, f'Outliers: {outlier_count}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, color='red', fontsize=8)
    if precision is not None:
        plt.text(0.85, 0.85, f'Precision: {precision:.3f}', horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=8)
        plt.text(0.85, 0.8, f'Recall: {recall:.3f}', horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=8)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        return fig


class ExtractSIFT(object):
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)
        self.num_kp = num_kp

    def run(self, img):
        img = img.astype(np.uint8)
        #    img = cv2.imread(img)
        cv_kp, desc = self.sift.detectAndCompute(img, None)

        kp = np.array([[_kp.pt[0], _kp.pt[1]] for _kp in cv_kp])  # N*2

        return kp[:self.num_kp], desc[:self.num_kp]  # 是直接按照scores降序排列的吗？需要验证！


def norm_kp(cx, cy, fx, fy, kp):
    # New kp
    kp = (kp - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
    return kp


def np_skew_symmetric(v):
    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M


def get_episym(x1, x2, dR, dt):
    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2) +
            1.0 / (Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2))

    return ys.flatten()


def get_gt(x1, x2, T):
    dR, dt = T[:3, :3], T[:3, 3]
    residual = get_episym(x1, x2, dR, dt)
    return residual
