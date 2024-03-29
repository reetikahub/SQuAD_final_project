"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF, QANet, QANet_extra
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD
from plot_attention_heatmap import plot_attention

def main(args):
    # Set up logging
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))


    # Get model
    # log.info('Building model...')
    # if args.model == 'bidaf':
    #     model = BiDAF(word_vectors=word_vectors,
    #                   char_vectors=char_vectors,
    #                   hidden_size=args.hidden_size,
    #                   drop_prob=0,
    #                   pos_size=args.pos_types,
    #                   pos_dim=args.pos_dim,
    #                   ner_size=args.ner_types,
    #                   ner_dim=args.ner_dim)
    # elif args.model == 'qanet':
    #     model = QANet(word_vec=word_vectors,
    #                   char_vec=char_vectors,
    #                   d_model=args.d_model,
    #                   drop_prob=0,
    #                   num_head=args.num_head)
    # else:
    #     model = QANet_extra(word_vec=word_vectors,
    #                         char_vec=char_vectors,
    #                         d_model=args.d_model,
    #                         drop_prob=0,
    #                         num_head=args.num_head,
    #                         pos_size=args.pos_types,
    #                         pos_dim=args.pos_dim,
    #                         ner_size=args.ner_types,
    #                         ner_dim=args.ner_dim)
    # model = nn.DataParallel(model, gpu_ids)

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.pos_types, args.ner_types, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = util.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs,  c_pos, c_ner, c_freq, c_em, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            c_pos = c_pos.to(device)
            c_ner = c_ner.to(device)
            c_freq = c_freq.to(device)
            c_em = c_em.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)
            
            load_paths = ['save/train/qanet_tags_pad-02', 'save/train/qanet_128_200_8-07', 'save/train/qanet_128h_4h-04', 'save/train/qanet_new_lr2_80-01']
            num_models = len(load_paths)
            #print(cw_idxs.shape)
            sz = cw_idxs.shape
            p1_sum = torch.zeros(sz).to(device)
            p2_sum = torch.zeros(sz).to(device)
            for load_path in load_paths:
                #Get parameters
                if load_path == 'save/train/qanet_tags_pad-02':
                    model_type = 'qanet_extra'
                    num_head = 8
                    d_model = 128
                    char_emb_file = 'data/char_emb2.json'
                elif load_path == 'save/train/qanet_128_200_8-07':
                    model_type = 'qanet'
                    num_head = 8
                    d_model = 128
                    char_emb_file = 'data/char_emb2.json'
                elif load_path == 'save/train/qanet_128h_4h-04':
                    model_type = 'qanet'
                    num_head = 4
                    d_model = 128
                    char_emb_file = 'data/char_emb.json'
                elif load_path == 'save/train/qanet_new_lr2_80-01':
                    model_type = 'qanet'
                    num_head = 1
                    d_model = 96
                    char_emb_file = 'data/char_emb.json'

                # Get embeddings
                #log.info('Loading embeddings...')
                word_vectors = util.torch_from_json(args.word_emb_file)
                char_vectors = util.torch_from_json(char_emb_file)
                load_path_full = join(load_path, 'best.pth.tar')
                # Get model
                if model_type == 'bidaf':
                    model = BiDAF(word_vectors=word_vectors, char_vectors=char_vectors, hidden_size=args.hidden_size, drop_prob=0, pos_size=args.pos_types,
                        pos_dim=args.pos_dim, ner_size=args.ner_types, ner_dim=args.ner_dim)
                elif model_type == 'qanet':
                    model = QANet(word_vec=word_vectors, char_vec=char_vectors, d_model=d_model, drop_prob=0, num_head=num_head)
                else:
                    model = QANet_extra(word_vec=word_vectors, char_vec=char_vectors, d_model=args.d_model, drop_prob=0, num_head=args.num_head,
                        pos_size=args.pos_types, pos_dim=args.pos_dim, ner_size=args.ner_types, ner_dim=args.ner_dim)
                model = nn.DataParallel(model, gpu_ids)
            #    log.info(f'Loading checkpoint from {load_path}...')
                model = util.load_model(model, load_path_full, gpu_ids, return_step=False)
                model = model.to(device)
                model.eval()
                # Forward
                if model_type == 'qanet':
                    log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
                else:
                    log_p1, log_p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs, c_pos, c_ner, c_freq, c_em)
                y1, y2 = y1.to(device), y2.to(device)
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                nll_meter.update(loss.item(), batch_size)
                # print(qw_idxs)
                # Get F1 and EM scores
                p1, p2 = log_p1.exp(), log_p2.exp()
                #print(p1[0,:5])
                #print(p2[0,:5])
                #print(p1.shape)
                p1_sum = p1_sum + p1
                p2_sum = p2_sum + p2
            p1_sum = p1_sum/num_models
            #print(p1_sum[0,:5])
            p2_sum = p2_sum/num_models
            #print(p2_sum[0,:5])
            #print(p1,p2)
            starts, ends = util.discretize(p1_sum, p2_sum, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = util.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)

            att = model.module.cq_att.get_attention_map()
            att = att.transpose(1, 2)
            file_name_const = join(args.save_dir, args.split)
            for i, qid in enumerate(ids.tolist()):
                file_name = file_name_const + str(qid) + 'attention.png'
                #print(cw_idxs[i])
                #print(qw_idxs[i])
                plot_attention(att[i].squeeze().cpu(), file_name, cw_idxs[i].squeeze(), qw_idxs[i].squeeze())

            #print(idx2pred, uuid2pred)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)
            dict = {}
            for qid in ids.tolist():
                uuid = gold_dict[str(qid)]["uuid"]
                dict[uuid] = p1
                dict[uuid] = p2
            # pred_dict.update(dict)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = util.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        att = att.transpose(1, 2)
        file_name_const = join(args.save_dir, args.split)
        for i, qid in enumerate(ids.tolist()):
            file_name = file_name_const + str(qid) + 'attention.png'
            #print(cw_idxs[i])
            #print(qw_idxs[i])
            plot_attention(att[i].squeeze().cpu(), file_name, cw_idxs[i].squeeze(), qw_idxs[i].squeeze())

        #print_att_heatmap(gold_dict, ids.tolist(), file_name, att.transpose(1, 2))

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        util.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])

    #Added by Reetika for further analysis
    sub_path = join(args.save_dir, args.split + '2' + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}2...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for id in sorted(pred_dict):
            csv_writer.writerow([id, pred_dict[id]])

if __name__ == '__main__':
    main(get_test_args())
