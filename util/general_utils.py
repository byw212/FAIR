import os
import torch
import inspect
from statistics import mean
from sklearn.metrics import silhouette_score,calinski_harabasz_score
import numpy as np
from datetime import datetime
from loguru import logger
from tqdm import tqdm
from copy import deepcopy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from util.cluster_and_log_utils import log_accs_from_preds

def get_feature(model,dataset,device):
    print('Prepare for extracting features')
    length = len(dataset)
    data_iter = iter(dataset)
    model.eval()
    with torch.no_grad():

        x, y, idx, z = next(data_iter)
        z = z[:, 0]
        uq_idx = idx.to(device)
        label = y.to(device)
        mask = z.to(device).bool()
        projs, features,_ = model(x.to(device))
        for i in range(1, length):
            x, y, idx, z = next(data_iter)
            z = z[:, 0].bool()
            proj, feat,_ = model(x.to(device))
            projs = torch.cat([projs, proj], dim=0)
            features = torch.cat([features, feat], dim=0)
            uq_idx = torch.cat([uq_idx, idx.to(device)], dim=0)
            label = torch.cat([label, y.to(device)], dim=0)
            mask = torch.cat([mask, z.to(device)], dim=0)
        sort_id, index = uq_idx.sort()
        transform = {int(i): num for num, i in enumerate(sort_id)}
        features = features[index]
        projs = projs[index]
        label = label[index]
        mask = mask[index]
        features = torch.nn.functional.normalize(features, dim=-1)
    information = (projs,features,label,mask,transform,index)
    print('Extract features successfully')


    return information

def get_mask(information,weights,uncentain_mask,device,epoch):
    features, label, mask, transform, index = information
    with torch.no_grad():
        logits = features@weights.T
        preds = torch.argmax(logits,dim=1)
        learn_mask = torch.zeros(features.shape[0]).to(device)
        logits = logits.T
        for i in range(len(weights)):
            label_i = (i==preds)
            size = label_i.sum()
            if size==0:
                #print(i)
                continue
            sim_i = logits[i][label_i]
            sort_i,_ = sim_i.sort()
            if i<98:
                th_low = sort_i[int(0.0 * size)]
            else:
                th_low = sort_i[int(0.0 * size)]

            choice1 = logits[i]>=th_low
            learn_mask += label_i * choice1# * uncentain_mask


    return transform, learn_mask.bool()

def prepare_training(epoch,information,args,mode= 'est', estimated_k = None,A=0.2):
    def transform(data):
        try:
            data = data.detach().cpu().numpy()
            return data
        except:
            return data
    projs,feats, targets, mask, transform, index = (transform(x) for x in information)

    if mode =='GT':
        noise = args.num_unlabeled_classes*A
        klist = [int(args.num_unlabeled_classes+noise+noise*np.cos(i)) for i in range(args.epochs)]
        klist[args.epochs-1] = args.num_unlabeled_classes

    if mode =='est':
        best_acc_k = estimated_k
        print('estimate K:', estimated_k)
        noise = (A - A * epoch / args.epochs) * (best_acc_k - args.num_labeled_classes)
        klist = [int(5+best_acc_k-args.num_labeled_classes+noise + noise * np.cos(i)) for i in range(args.epochs)]


    linked = linkage(feats, method="ward")
    dist = linked[:, 2][:-args.num_labeled_classes]
    K = int(klist[epoch])
    print('estimate K*1.X:',K+args.num_labeled_classes)
    d = dist[-K]
    preds = fcluster(linked, t=d, criterion='distance')
    best_acc_k = max(preds)
    class_feat = [[] for _ in range(best_acc_k)]
    reorder_w = [0 for _ in range(best_acc_k)]
    preds = preds.astype(int)
    targets = targets.astype(int)
    for num, i in enumerate(preds):
        class_feat[i - 1].append(feats[num].reshape(1, -1))
    class_feat = [np.concatenate(x) for x in class_feat]
    pseudo_w = [x.mean(axis=0).reshape(1, -1) for x in class_feat]
    w = np.concatenate(pseudo_w)
    logits = feats @ w.T
    preds1 = np.argmax(logits, axis=-1)
    uncentain_mask = preds == (preds1 + 1)
    uncentain_mask = torch.tensor(uncentain_mask)
    print(uncentain_mask.sum() / len(logits))



    D = max(preds.max(), args.num_labeled_classes) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        if mask[i]:
            w[preds[i], targets[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}
    flag = []
    for i in range(args.num_labeled_classes):
        # f = feats[(targets==i)*mask].mean(0)
        # reorder_w[i] = f.reshape(1,-1)
        reorder_w[i] = pseudo_w[ind_map[i] - 1]
        flag.append(ind_map[i] - 1)
    flag.sort()
    for i in reversed(flag):
        pseudo_w.pop(i)
    for i in range(args.num_labeled_classes, best_acc_k):

        reorder_w[i] = pseudo_w[i-args.num_labeled_classes]

    w = np.concatenate(reorder_w)

    return best_acc_k, w,uncentain_mask

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args, runner_name=None, exp_id=None):
    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Either generate a unique experiment ID, or use one which is passed
    if exp_id is None:

        if args.exp_name is None:
            raise ValueError("Need to specify the experiment name")
        # Unique identifier for experiment
        now = '{}_({:02d}_{:02d}_'.format(args.exp_name, datetime.now().day, datetime.now().month) + \
              datetime.now().strftime("%H_%M_%S") + ')'

        log_dir = os.path.join(root_dir, 'log', now)
        while os.path.exists(log_dir):
            now = '({:02d}_{:02d}_'.format(datetime.now().day, datetime.now().month) + \
                  datetime.now().strftime("%H_%M_%S") + ')'

            log_dir = os.path.join(root_dir, 'log', now)

    else:

        log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')

    print(f'Experiment saved to: {args.log_dir}')

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    print(runner_name)
    print(args)

    return args


class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, generator=None):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor =  self.rank + rand_tensor * self.num_replicas
        yield from iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples
