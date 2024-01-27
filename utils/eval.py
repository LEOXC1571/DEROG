import numpy as np
import torch


def eval_data_preprocess(y, raw_pred, mask, task):
    if task == 'Binary classification':
        pred_prob = raw_pred.sigmoid()
        if y.shape[1] > 1:
            # multi-task
            preds = []
            targets = []
            for i in range(y.shape[1]):
                # pred and target per task
                preds.append(pred_prob[:, i][mask[:, i]].detach().cpu().numpy())
                targets.append(y[:, i][mask[:, i]].detach().cpu().numpy())
            return preds, targets
        pred = pred_prob[mask].reshape(-1).detach().cpu().numpy()
    elif task == 'Multi-label classification':
        pred_prob = raw_pred.softmax(dim=1)
        pred = pred_prob[mask].detach().cpu().numpy()
    elif 'Regression' in task:
        pred = raw_pred[mask].reshape(-1).detach().cpu().numpy()
    else:
        raise ValueError('Dataset task value error.')

    target = y[mask].reshape(-1).detach().cpu().numpy()

    return pred, target

def eval_data_postprocess(pred, label, task):
    pred, label = np.concatenate(pred), np.concatenate(label)
    if task == 'Binary classification':
        pred = np.round(pred)
        pred_true = (pred == label)
    elif task == 'Multi-label classification':
        pred = np.argmax(pred, axis=1)
        pred_true = (pred == label)
    else:
        return pred, None
    return pred, pred_true

def eval_score(pred_all, target_all, score_func):

    np.seterr(invalid='ignore')

    assert type(pred_all) is list, 'Wrong prediction input.'
    if type(pred_all[0]) is list:
        # multi-task
        all_task_preds = []
        all_task_targets = []
        for task_i in range(len(pred_all[0])):
            preds = []
            targets = []
            for pred, target in zip(pred_all, target_all):
                preds.append(pred[task_i])
                targets.append(target[task_i])
            all_task_preds.append(np.concatenate(preds))
            all_task_targets.append(np.concatenate(targets))

        scores = []
        for i in range(len(all_task_preds)):
            if all_task_targets[i].shape[0] > 0:
                scores.append(np.nanmean(score_func(all_task_targets[i], all_task_preds[i])))
        score = np.nanmean(scores)
    else:
        pred_all = np.concatenate(pred_all)
        target_all = np.concatenate(target_all)
        score = np.nanmean(score_func(target_all, pred_all))
    return score
