import os
import torch
import inspect
from statistics import mean
from sklearn.metrics import silhouette_score
import numpy as np
from datetime import datetime
from loguru import logger
from tqdm import tqdm
from copy import deepcopy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from util.cluster_and_log_utils import log_accs_from_preds
def get_features(model,weights,dataset,device,uncentain_mask):
    length = len(dataset)
    data_iter = iter(dataset)
    with torch.no_grad():

        x, y, idx, z = next(data_iter)
        z = z[:, 0]
        uq_idx = idx.to(device)
        label = y.to(device)
        mask = z.to(device).bool()
        _,features = model(x.to(device))
        logits = features@weights.T
        preds = torch.argmax(logits,dim=1)

        for i in range(1, length):
            x, y, idx, z = next(data_iter)
            z = z[:, 0].bool()
            _,feat = model(x.to(device))
            features = torch.cat([features, feat], dim=0)
            logit = features@weights.T
            logits = torch.cat([logits, logit], dim=0)
            preds = torch.cat([preds, torch.argmax(logit,dim=1)], dim=0)
            uq_idx = torch.cat([uq_idx,idx.to(device)], dim=0)
            label = torch.cat([label, y.to(device)], dim=0)
            mask = torch.cat([mask, z.to(device)], dim=0)
        sort_id,index = uq_idx.sort()
        transform = {int(i):num for num,i in enumerate(sort_id)}
        features = features[index]
        label = label[index]
        logits = logits[index]
        logits = logits.T
        mask = mask[index]
        preds = preds[index]
        features = torch.nn.functional.normalize(features, dim=-1)
        mask1 = torch.zeros(features.shape[0]).to(device)
        mask2 = torch.zeros(features.shape[0]).to(device)

        n=0
        for i in range(len(weights)):
            label_i = (i==preds)
            size = label_i.sum()
            n+=size
            if size==0:
                #print(i)
                continue
            sim_i = logits[i][label_i]
            sort_i,_ = sim_i.sort()
            if i<100:
                th_low = sort_i[int(0.1 * size)]
            else:
                th_low = sort_i[int(0.05 * size)]

            choice1 = logits[i]>=th_low
            mask1 += label_i*choice1#*uncentain_mask


    return transform, mask1.bool()


def prepare_trainingZ_(epoch,model,loader,args):
    with torch.no_grad():
        #model.eval()
        idxs = np.array([])
        all_feats_val = []
        targets = np.array([])
        mask = np.array([])
        num_train_classes = len(args.train_classes)

        for batch_idx, batch in enumerate(tqdm(loader)):
            images, label, idx,z= batch
            images = images.cuda()
            z = z[:, 0].bool()
            _,featuress = model(images)
            detach_feat = featuress.detach().cpu().numpy()
            all_feats_val.append(detach_feat)
            targets = np.append(targets, label.cpu().numpy())
            idxs = np.append(idxs, idx.cpu().numpy())

            #mask = np.append(mask, z.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                             else False for x in label]))
    best_acc = 0
    idxs = torch.tensor(idxs)
    _, index = idxs.sort()
    mask = mask.astype(bool)
    feats = np.concatenate(all_feats_val)
    klist = [225 + 25 * np.cos(i) for i in range(200)]
    # print('Estimate of k: ',best_acc_k)
    # K = int((best_acc_k - args.num_labeled_classes) * 2)
    K = int(klist[epoch])

    kmeans = KMeans(n_clusters=K, random_state=0).fit(feats)
    preds = kmeans.labels_
    centers = kmeans.cluster_centers_


    best_acc_k = max(preds)
    reorder_w = [0 for _ in range(best_acc_k)]
    preds = preds.astype(int)
    targets = targets.astype(int)
    norm = np.linalg.norm(centers,axis=-1)
    centers = centers /norm.reshape(-1,1)
    D = max(preds.max(), num_train_classes) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        if mask[i]:
            w[preds[i], targets[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}
    flag = []
    for i in range(num_train_classes):
        #f = feats[(targets==i)*mask].mean(0)
        #reorder_w[i] = f.reshape(1,-1)
        reorder_w[i] = centers[ind_map[i]].reshape(1,-1)
        flag.append(ind_map[i])
    flag.sort()
    for i in reversed(flag):
        centers = np.delete(centers, i,axis=0)
    for i in range(num_train_classes, best_acc_k):
        reorder_w[i] = centers[i-num_train_classes].reshape(1,-1)

    w = np.concatenate(reorder_w)

    return best_acc_k, w

def prepare_training(counter,model,loader,args):

    with torch.no_grad():
        #model.eval()
        idxs = np.array([])
        all_feats_val = []
        targets = np.array([])
        mask = np.array([])
        num_train_classes = len(args.train_classes)

        for batch_idx, batch in enumerate(tqdm(loader)):
            images, label, idx,z= batch
            images = images.cuda()
            z = z[:, 0].bool()
            _,featuress = model(images)
            detach_feat = featuress.detach().cpu().numpy()
            all_feats_val.append(detach_feat)
            targets = np.append(targets, label.cpu().numpy())
            idxs = np.append(idxs, idx.cpu().numpy())

            mask = np.append(mask, z.cpu().numpy())
            #mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
            #                                 else False for x in label]))
    best_acc = 0
    idxs = torch.tensor(idxs)
    _, index = idxs.sort()
    mask = mask.astype(bool)
    feats = np.concatenate(all_feats_val)

    linked = linkage(feats, method="ward")

    dist = linked[:, 2][:-args.num_labeled_classes]
    tolerance = 0
    best_sore = -1
    for d in reversed(dist):
        preds = fcluster(linked, t=d, criterion='distance')
        k = max(preds)
        silhouette_avg = silhouette_score(feats,preds)
        all_acc_test, old_acc_test, new_acc_test = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                                       T=0, eval_funcs=args.eval_funcs,
                                                                       save_name='')

        if silhouette_avg> best_sore:  # save best labeled acc without knowing GT K
            best_acc = old_acc_test
            best_sore = silhouette_avg
            best_acc_k = k
            best_acc_d = d
            tolerance = 0
        else:
            tolerance += 1

        if tolerance == 50:
            break

    klist = [int(120+25*np.cos(i)) for i in range(200)]
    print('Estimate of k: ',best_acc_k)
    #K = int((best_acc_k - args.num_labeled_classes) * 2)
    K = int(klist[counter])
    print('Estimate of k * 1.X: ', args.num_labeled_classes+K)
    d =  dist[-K]
    preds = fcluster(linked, t= d, criterion='distance')
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
    #norm = np.linalg.norm(w,axis=-1)
    #w = w /norm.reshape(-1,1)

    logits = feats@w.T
    preds1 = np.argmax(logits,axis=-1)
    # for i in range(len(preds)):
    #     if preds[i]!=(preds1[i]+1):
    #         print(logits[i][int(preds[i]-1)],logits[i][int(preds1[i])])
    #         print(preds1[i])

    #
    uncentain_mask = preds==(preds1+1)
    uncentain_mask = torch.tensor(uncentain_mask)
    uncentain_mask =uncentain_mask[index]

    # print(pseudo_w.shape)
    D = max(preds.max(), num_train_classes) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        if mask[i]:
            w[preds[i], targets[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}
    flag = []
    num = 6000
    for i in range(num_train_classes):
        #f = feats[(targets==i)*mask].mean(0)
        #reorder_w[i] = f.reshape(1,-1)
        reorder_w[i] = pseudo_w[ind_map[i] - 1]
        flag.append(ind_map[i] - 1)
    flag.sort()
    for i in reversed(flag):
        pseudo_w.pop(i)
    for i in range(num_train_classes, best_acc_k):

        reorder_w[i] = pseudo_w[i-num_train_classes]

    w = np.concatenate(reorder_w)

    return best_acc_k, w#,uncentain_mask

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
