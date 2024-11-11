import argparse
from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import dino_pretrain_path
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
import vision_transformer as vits
from util.general_utils import AverageMeter, init_experiment, prepare_training, get_feature,get_mask
from util.cluster_and_log_utils import log_accs_from_preds, test_agglo
from config import exp_root
from torch.utils.data import Subset
from model import DINOHead, Classifier, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, \
    get_params_groups



def pretrain(student, train_loader,test_loader, args):
    for name, m in student[0].named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= 6:
                m.requires_grad = True
    regularized = []
    not_regularized = []
    for name, param in student[0].named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    params_groups1 =  [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    regularized1 = []
    not_regularized1 = []
    for name, param in student[1].named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized1.append(param)
        else:
            regularized1.append(param)
    params_groups2 =  [{'params': regularized1}, {'params': not_regularized1, 'weight_decay': 0.}]
    lr1 = 0.01
    lr2 = 0.05

    optimizer1 = SGD(params_groups1, lr=lr1, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer2 = SGD(params_groups2, lr=lr2, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler1 = lr_scheduler.CosineAnnealingLR(
        optimizer1,
        T_max=args.pratrain_epochs,
        eta_min=lr1* 1e-3,
    )
    exp_lr_scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer2,
        T_max=args.pratrain_epochs,
        eta_min=lr2 * 1e-3,
    )
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.pratrain_epochs):
        with torch.no_grad():
            student.eval()
            all_feats = []
            labels = []
            ids = []
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                images, class_labels, uq_idxs = batch
                images = images.to(device)
                _,features,_ = student(images)
                ids.append(uq_idxs)
                labels.append(class_labels)
                all_feats.append(features.detach().cpu())
            ids = torch.cat(ids,dim=0)
            sort_id,index = ids.sort()
            transform = {int(i):num for num,i in enumerate(sort_id)}
            all_feats = torch.cat(all_feats)
            all_feats = all_feats.to(device)
            all_feats = nn.functional.normalize(all_feats, dim=-1, p=2)
            all_feats = all_feats[index]
            labels = torch.cat(labels,dim=0)
            labels = labels.to(device)
            labels = labels[index]
        loss_record = AverageMeter()
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs = batch
            uq_idxs = torch.tensor([transform[int(i)] for i in uq_idxs]).to(device)
            class_labels = class_labels.to(device)
            #class_labels = torch.cat([class_labels,class_labels],dim=0)
            images = images.to(device)
            past_feat = all_feats[uq_idxs]


            with (torch.cuda.amp.autocast(fp16_scaler is not None)):
                student_proj , x , logits = student(images)
                student_proj  = nn.functional.normalize(student_proj, dim=-1, p=2)
                sim = torch.diag(x@past_feat.T)
                cls_loss = nn.CrossEntropyLoss()(logits/0.1, class_labels)
                contrastive_loss = 0
                contrastive_labels = torch.zeros(1,dtype=torch.long).to(device)
                for i in range(len(student_proj)):

                    nagetive_logit = x[i].view(1,-1)@all_feats[labels==class_labels[i]].T
                    positve_logit = sim[i].view(1,-1)
                    logit = torch.cat([positve_logit,nagetive_logit],dim=-1)
                    contrastive_loss += torch.nn.CrossEntropyLoss()(logit/0.1, contrastive_labels)
                contrastive_loss  = contrastive_loss/len(student_proj)


                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '

                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0.5*cls_loss+1.0*contrastive_loss
                #loss += ploss


            # Train ac
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer1.step()
                optimizer2.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer1)
                fp16_scaler.step(optimizer2)

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        exp_lr_scheduler1.step()
        exp_lr_scheduler2.step()
        save_dict = {
            'model': student.state_dict()
        }

        torch.save(save_dict, args.model_path)
        #pra_test(student, test_loader)
    return student


def train(student, train_loader, init_dataset, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )
    best_test_acc_lab=0

    for epoch in range(args.epochs):
        length = len(train_dataset)
        indices = list(range(length))
        # samples = np.random.choice(indices, size=20000, replace=False)
        # sub_init_dataset = Subset(init_dataset, samples)
        init_loader = DataLoader(init_dataset, num_workers=args.num_workers,
                                 batch_size=512, shuffle=False, pin_memory=False)
        information = get_feature(model, init_loader, device)
        loss_record = AverageMeter()
        student.train()

        if (epoch % 10 == 0 and epoch<=150) or epoch<200:
            pred_K, weights,uncentain_mask = prepare_training(epoch,information,args,mode='GT')
            uncentain_mask = uncentain_mask.to(device)
            weights = torch.tensor(weights).to(device)
            weights = torch.nn.functional.normalize(weights, dim=-1)
            weights.requires_grad = True
            optimizer_w = SGD([weights], lr=5e-3, momentum=args.momentum,
                              weight_decay=args.weight_decay)
            w_exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer_w,
                T_max=10,
                eta_min=5e-3
            )

        #transform,learn_mask1= get_mask(information,weights,uncentain_mask,device,epoch)
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            #uq_idxs = torch.tensor([transform[int(i)] for i in uq_idxs])
            #sub_learn_mask1 = learn_mask1[uq_idxs]
            #sub_learn_mask2 = learn_mask2[uq_idxs]
            #double_learn_mask1 = torch.cat([sub_learn_mask1, sub_learn_mask1], dim=0)

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(device)

            with (torch.cuda.amp.autocast(fp16_scaler is not None)):
                student_proj, features,_ = student(images)


                student_out = features @ weights.T
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                d_mask_lab = torch.cat([mask_lab,mask_lab],dim=0)

                # clustering, unsup

                cluster_loss = 0

                cluster_loss1= cluster_criterion(student_out[~d_mask_lab],
                                                  teacher_out[~d_mask_lab], epoch)
                cluster_loss += cluster_loss1


                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj,temperature=1.0,device=device)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                if cluster_loss!=0:
                    pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                #loss += ploss


            # Train ac
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            optimizer_w.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
                optimizer_w.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update(optimizer_w)

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))


        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')

        all_acc, old_acc, new_acc = test(student, weights, unlabelled_train_loader, epoch=epoch,
                                         save_name='Train ACC Unlabelled', args=args)
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()
        w_exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if all_acc > best_test_acc_lab:

            torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

            # inductive
            best_test_acc_lab = all_acc
            # transductive
            best_train_acc_lab = old_acc
            best_train_acc_ubl = new_acc
            best_train_acc_all = all_acc

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def pra_test(model, test_loader):
    model.eval()

    preds, targets,mask = [], [],[]
    #mask = np.array([])
    for batch_idx, (images, label,_) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            _, _,logits = model(images)

            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    print((preds[mask]==targets[mask]).sum()/len(targets[mask]))


def test(model, classifier, test_loader, epoch, save_name, args):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            _, features,_ = model(images)
            logits = features @ classifier.T

            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask,
                             np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub',
                        help='options: cifar10, cifar100, imagenet_100, cub, scars,aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)

    parser.add_argument('--memax_weight', type=float, default=1)
    parser.add_argument('--pratrain_epochs', type=int, default=50)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float,
                        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float,
                        help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--vector_dim', default=256, type=int)
    parser.add_argument('--exp_name', default='test_component1', type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[args.dataset_name])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = vits.__dict__['vit_base']()
    state_dict = torch.load(dino_pretrain_path, map_location='cpu')
    backbone.load_state_dict(state_dict)

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)

    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len/unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    finetune_dataset = deepcopy(train_dataset.labelled_dataset)
    finetune_dataset.transform = test_transform

    finetune_loader = DataLoader(finetune_dataset, num_workers=args.num_workers,
                             batch_size=64, shuffle=True, pin_memory=False)

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)
    init_dataset = deepcopy(train_dataset)
    init_dataset.unlabelled_dataset.transform = test_transform
    init_dataset.labelled_dataset.transform = test_transform


    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)

    model = pretrain(model,finetune_loader, test_loader_labelled,args)
    for m in model[0].parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model[0].named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    train(model, train_loader, init_dataset, test_loader_labelled, test_loader_unlabelled, args)
