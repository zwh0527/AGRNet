import torch
import numpy as np
import os
import cv2
from tqdm import tqdm
from loss import MatchLoss
from evaluation import eval_nondecompose, eval_decompose, pose_auc
from utils import tocuda, get_pool_result


def test_sample(args):
    _xs, _dR, _dt, _e_hat, _y_hat, _y_gt, config, = args
    _xs = _xs.reshape(-1, 4).astype('float64')
    _dR, _dt = _dR.astype('float64').reshape(3, 3), _dt.astype('float64')
    _y_hat_out = _y_hat.flatten().astype('float64')
    e_hat_out = _e_hat.flatten().astype('float64')

    _x1 = _xs[:, :2]
    _x2 = _xs[:, 2:]
    # current validity from network
    _valid = _y_hat_out
    # choose top ones (get validity threshold)
    _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
    _mask_before = _valid >= max(0, _valid_th)

    if not config.use_ransac:
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_nondecompose(_x1, _x2, e_hat_out, _dR, _dt, _y_hat_out)
    else:
        # actually not use prob here since probs is None
        _err_q, _err_t, _, _, _num_inlier, _mask_updated, _R_hat, _t_hat = \
            eval_decompose(_x1, _x2, _dR, _dt, mask=_mask_before, method=cv2.RANSAC, \
                           probs=None, weighted=False, use_prob=True)
    if _R_hat is None:
        _R_hat = np.random.randn(3, 3)
        _t_hat = np.random.randn(3, 1)

    # calculate the inlier ratio
    _inlier_ratio = np.mean((_y_gt.reshape(-1) < config.obj_geod_th).astype(float))

    return [float(_err_q), float(_err_t), float(_num_inlier), _R_hat.reshape(1, -1), _t_hat.reshape(1, -1),
            float(_inlier_ratio)]


def dump_res(res_path, eval_res, use_ransac=False, mode='test'):
    # dump test results
    err_q = np.array(eval_res["err_q"]) * 180.0 / np.pi
    err_t = np.array(eval_res["err_t"]) * 180.0 / np.pi
    err_qt = np.maximum(err_q, err_t)
    inlier_ratio = eval_res['inlier_ratio']

    # sort according to inlier ratio
    sort_idx = np.argsort(inlier_ratio)
    inlier_ratio = inlier_ratio[sort_idx]
    err_qt = err_qt[sort_idx]
    if mode == 'test':
        P, R, F = eval_res['P'][sort_idx], eval_res['R'][sort_idx], eval_res['F'][sort_idx]

    # dump res under different thresholds of inlier ratios
    rng = [0.05, 0.10, 0.15, 1.00] if mode == 'test' else [1.00]
    for thr in rng:
        print(f'\ninlier ratio is {thr}')
        ir_idx = np.searchsorted(inlier_ratio, thr)

        cur_err_qt = err_qt[:ir_idx + 1]
        sort_idx = np.argsort(cur_err_qt)
        cur_err_qt = np.array(cur_err_qt.copy())[sort_idx]

        # real aucs borrowed from CLNet
        aucs = pose_auc(cur_err_qt, [5, 10, 20])
        aucs = [100. * yy for yy in aucs]
        print('Evaluation Results (mean over {} pairs):'.format(len(cur_err_qt)))
        print('AUC@5\t AUC@10\t AUC@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2]))
        filename = f'/aucs_ransac_inlierRatio={thr}.txt' if use_ransac else f'/aucs_dlt_inlierRatio={thr}.txt'
        np.savetxt(res_path + filename, np.asarray(aucs))

        # accs
        accs = np.searchsorted(cur_err_qt, [5, 10, 15, 20]) / len(cur_err_qt) * 100.
        print('ACC@5\t ACC@10\t ACC@15\t ACC@20\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(accs[0], accs[1], accs[2], accs[3]))
        filename = f'/accs_ransac_inlierRatio={thr}.txt' if use_ransac else f'/accs_dlt_inlierRatio={thr}.txt'
        np.savetxt(res_path + filename, np.asarray(accs))

        # fake aucs
        aucs_fake = [np.mean(accs[:(i + 1)]) for i in range(4)]
        print('AUC@5\t AUC@10\t AUC@15\t AUC@20\t (fake)')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs_fake[0], aucs_fake[1], aucs_fake[2], aucs_fake[3]))
        filename = f'/aucs_fake_ransac_inlierRatio={thr}.txt' if use_ransac else f'/aucs_fake_dlt_inlierRatio={thr}.txt'
        np.savetxt(res_path + filename, np.asarray(aucs_fake))

        if mode == 'test':
            # P, R, F
            cur_P, cur_R, cur_F = np.nanmean(P[:ir_idx + 1]), np.nanmean(R[:ir_idx + 1]), np.nanmean(F[:ir_idx + 1])
            print('P\t R\t F\t')
            print('{:.4f}\t {:.4f}\t {:.4f}\t'.format(cur_P, cur_R, cur_F))
            filename = f'/PRF_inlierRatio={thr}.txt'
            np.savetxt(res_path + filename, np.asarray([cur_P, cur_R, cur_F]))

    # Return real auc10 in overall sequences
    ret_val = aucs[1]
    return ret_val


def denorm(x, T):
    x = (x - np.array([T[0, 2], T[1, 2]])) / np.asarray([T[0, 0], T[1, 1]])
    return x


def test_process(mode, model, cur_global_step, data_loader, config):
    model.eval()
    match_loss = MatchLoss(config)
    loader_iter = iter(data_loader)

    # save info given by the network
    network_infor_list = ["geo_losses", "cla_losses", "l2_losses", 'precisions', 'recalls', 'f_scores']
    network_info = {info: [] for info in network_infor_list}

    results, pool_arg = [], []
    eval_step, eval_step_i, num_processor = 100, 0, 8
    with torch.no_grad():
        loader_iter = tqdm(loader_iter)
        for test_data in loader_iter:
            test_data = tocuda(test_data)
            res_logits, res_e_hat = model(test_data)
            y_hat, e_hat = res_logits[-1], res_e_hat[-1]
            loss, geo_loss, cla_loss, l2_loss, prec, rec = match_loss.run(cur_global_step, test_data, y_hat, e_hat)
            info = [geo_loss, cla_loss, l2_loss, prec, rec, 2 * prec * rec / (prec + rec + 1e-15)]
            for info_idx, value in enumerate(info):
                network_info[network_infor_list[info_idx]].append(value)

            if config.use_fundamental:
                # unnorm F
                e_hat = torch.matmul(torch.matmul(test_data['T2s'].transpose(1, 2), e_hat.reshape(-1, 3, 3)),
                                     test_data['T1s'])
                # get essential matrix from fundamental matrix
                e_hat = torch.matmul(torch.matmul(test_data['K2s'].transpose(1, 2), e_hat.reshape(-1, 3, 3)),
                                     test_data['K1s']).reshape(-1, 9)
                e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)

            for batch_idx in range(e_hat.shape[0]):
                test_xs = test_data['xs'][batch_idx].detach().cpu().numpy()
                if config.use_fundamental:  # back to original
                    x1, x2 = test_xs[0, :, :2], test_xs[0, :, 2:4]
                    T1, T2 = test_data['T1s'][batch_idx].cpu().numpy(), test_data['T2s'][batch_idx].cpu().numpy()
                    x1, x2 = denorm(x1, T1), denorm(x2, T2)  # denormalize coordinate
                    K1, K2 = test_data['K1s'][batch_idx].cpu().numpy(), test_data['K2s'][batch_idx].cpu().numpy()
                    x1, x2 = denorm(x1, K1), denorm(x2, K2)  # normalize coordiante with intrinsic
                    test_xs = np.concatenate([x1, x2], axis=-1).reshape(1, -1, 4)

                pool_arg += [(test_xs, test_data['Rs'][batch_idx].detach().cpu().numpy(), \
                              test_data['ts'][batch_idx].detach().cpu().numpy(),
                              e_hat[batch_idx].detach().cpu().numpy(), \
                              y_hat[batch_idx].detach().cpu().numpy(), \
                              test_data['ys'][batch_idx, :, 0].detach().cpu().numpy(), config)]

                eval_step_i += 1
                if eval_step_i % eval_step == 0:
                    results += get_pool_result(num_processor, test_sample, pool_arg)
                    pool_arg = []

        if len(pool_arg) > 0:
            results += get_pool_result(num_processor, test_sample, pool_arg)

    measure_list = ["err_q", "err_t", "num", 'R_hat', 't_hat', 'inlier_ratio']
    eval_res = {}
    for measure_idx, measure in enumerate(measure_list):
        eval_res[measure] = np.asarray([result[measure_idx] for result in results])
    if mode == 'test':
        eval_res['P'] = np.asarray(network_info['precisions'])
        eval_res['R'] = np.asarray(network_info['recalls'])
        eval_res['F'] = np.asarray(network_info['f_scores'])

    if config.res_path == '':
        config.res_path = os.path.join(config.log_path[:-5], mode)
    ret_val = dump_res(config.res_path, eval_res, config.use_ransac, mode)
    return [ret_val, np.nanmean(np.asarray(network_info['geo_losses'])),
            np.nanmean(np.asarray(network_info['cla_losses'])), \
            np.nanmean(np.asarray(network_info['l2_losses'])), np.nanmean(np.asarray(network_info['precisions'])), \
            np.nanmean(np.asarray(network_info['recalls'])), np.nanmean(np.asarray(network_info['f_scores']))]


def test(data_loader, model, config):
    save_file_best = os.path.join(config.model_path, 'model_best.pth')
    if not os.path.exists(save_file_best):
        print("Model File {} does not exist! Quiting".format(save_file_best))
        exit(1)
    # Restore model
    checkpoint = torch.load(save_file_best)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    print("Restoring from " + str(save_file_best) + ', ' + str(start_epoch) + "epoch...\n")
    if config.res_path == '':
        config.res_path = config.model_path[:-5] + 'test'
    print('save result to ' + config.res_path)
    va_res = test_process("test", model, 0, data_loader, config)
    print('test result ' + str(va_res))


def valid(data_loader, model, step, config):
    config.use_ransac = False
    return test_process("valid", model, step, data_loader, config)
