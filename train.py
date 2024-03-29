"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
import math

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, QANet, QANet_extra
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    # Added by Reetika
    char_vectors = util.torch_from_json(args.char_emb_file)
    # Get model
    log.info('Building model...')
    # Added by Reetika - char_vectors here and hard-coded drop_prob_char - can be added to arguments later,
    # # Also added len_c and len_q arguments

    if args.model == 'bidaf':
        model = BiDAF(word_vectors=word_vectors,
                    char_vectors=char_vectors,
                    hidden_size=args.hidden_size,
                    drop_prob=args.drop_prob,
                    pos_size=args.pos_types,
                    pos_dim=args.pos_dim,
                    ner_size=args.ner_types,
                    ner_dim=args.ner_dim)
    elif args.model == 'qanet':
        model = QANet(word_vec=word_vectors,
                        char_vec=char_vectors,
                        d_model=args.d_model,
                        drop_prob=args.drop_prob,
                        num_head=args.num_head)
    else:
        model = QANet_extra(word_vec=word_vectors,
                        char_vec=char_vectors,
                        d_model=args.d_model,
                        drop_prob=args.drop_prob,
                        num_head=args.num_head,
                        pos_size=args.pos_types, pos_dim=args.pos_dim, ner_size=args.ner_types, ner_dim=args.ner_dim)

    model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    #parameters = filter(lambda param: param.requires_grad, model.parameters())
    #optimizer = optim.Adam(lr=args.lr, betas=(0.8, 0.999), eps=1e-7, weight_decay=args.l2_wd, params=model.parameters())
    #lr = args.lr
    #m = 1 / math.log2(1e3)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: m * math.log2(ee + 1) if ee < 1e3 else 1)

    if args.model == 'bidaf':
        optimizer = optim.Adadelta(model.parameters(), args.lr, weight_decay=args.l2_wd)
        scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = optim.Adam(lr=args.lr, betas=(0.8, 0.999), eps=1e-7, weight_decay=args.l2_wd, params=parameters)
        m = 1 / math.log2(1e3)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ee: m * math.log2(ee + 1) if ee < 1e3 else 1)

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.pos_types, args.ner_types, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file,  args.pos_types, args.ner_types, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, c_pos, c_ner, c_freq, c_em, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                c_pos = c_pos.to(device)
                c_ner = c_ner.to(device)
                c_freq = c_freq.to(device)
                c_em = c_em.to(device)
                # Added by Reetika
                cc_idxs = cc_idxs.to(device)
                qc_idxs = qc_idxs.to(device)

                batch_size = cw_idxs.size(0)
                optimizer.zero_grad()

                # Forward
                # Added by Reetika - character level context and question in model
                if args.model == 'qanet':
                    log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                else:
                    log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs, c_pos, c_ner, c_freq, c_em)    
                # print("max of log p1:{}{}{}".format(log_p1[0,57], log_p1[1,45], log_p1[2,113]))
                # print("max of log p2:{}{}{}".format(log_p2[0,60], log_p2[1,47], log_p2[2,113]))
                y1, y2 = y1.to(device), y2.to(device)
                # print("max of y1:{}".format(y1))
                # print("max of y2:{}".format(y2))
                # print("y1 loss:{}".format(F.nll_loss(log_p1, y1)))
                # print("y2 loss:{}".format(F.nll_loss(log_p2, y2)))

                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                loss_val = loss.item()

                # Backward
                loss.backward()
                # print(model.parameters())
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                #scheduler.step(step // batch_size)
                scheduler.step()
                ema(model, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results, pred_dict = evaluate(model, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2,
                                                  args.model)
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2, model_type):
    nll_meter = util.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, c_pos, c_ner, c_freq, c_em, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward

            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            c_pos = c_pos.to(device)
            c_ner = c_ner.to(device)
            c_freq = c_freq.to(device)
            c_em = c_em.to(device)
            # Added by Reetika
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)

            batch_size = cw_idxs.size(0)

            # Forward
            # Added by Reetika - character level context and question in model
            if model_type == 'qanet':
                log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
            else:
                log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs, c_pos, c_ner, c_freq, c_em)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)
            # print(starts, ends)
            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model.train()
    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
