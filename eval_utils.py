import numpy as np

def calc_score_variance(true_labels, var_fs):
    uncertainties = var_fs 
    assert true_labels.shape[0] == uncertainties.shape[0]

    binary_labels = np.where(true_labels > 100, 1, 0) #live: 0, spoof: 1
    tnr95, da, roc = calc_paper_results(binary_labels, uncertainties) 
    print("TNR95: {}\nDA   : {}\nROC  : {}".format(tnr95, da, roc))
    return binary_labels, uncertainties, [tnr95, da, roc]


def calc_auroc(binary_labels, uncertainties):
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(binary_labels, uncertainties)
    auc = metrics.auc(fpr, tpr)
    return auc

def calc_paper_results(binary_labels, uncertainties):
    N = np.shape(binary_labels)[0]
    auroc = calc_auroc(binary_labels, uncertainties)

    order = np.argsort(uncertainties)
    uncertainties = uncertainties[order]
    binary_labels = binary_labels[order]
    print(N)

    fnrs = np.empty(N)
    fprs = np.empty(N)
    tprs = np.empty(N)
    for T in range(1, N+1):
        fp = np.sum(binary_labels[:T])
        tp = T - fp
        tn = np.sum(binary_labels[T:])
        fn = (N - T) - tn
        fnrs[T-1] = fn / (fn + tp)
        fprs[T-1] = fp / (fp + tn)
        tprs[T-1] = tp / (tp + fn)
    idx95TPR = np.argmin(np.where(tprs < 0.95, 100, tprs))
    tnr95tpr = 1.0 - fprs[idx95TPR]
    d_acc = 1.0 - np.amin(np.add(fprs, fnrs) * 0.5) # live:spoof = 50:50

    return tnr95tpr, d_acc, auroc
