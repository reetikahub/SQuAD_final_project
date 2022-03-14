import numpy as np
import torch
data = np.load('data/train2.npz')
# np.savez('data/smaller_train2.npz',
#          context_idxs=data['context_idxs'][:3],
#          context_char_idxs=data['context_char_idxs'][:3],
#          context_pos_tags=data['context_pos_tags'][:3],
#          context_ner_tags=data['context_ner_tags'][:3],
#          context_freq_tags=data['context_freq_tags'][:3],
#          context_em_tags=data['context_em_tags'][:3],
#          ques_idxs=data['ques_idxs'][:3],
#          ques_char_idxs=data['ques_char_idxs'][:3],
#          y1s=data['y1s'][:3],
#          y2s=data['y2s'][:3],
#          ids=data['ids'][:3])
# print(data['context_idxs'][0])
# print(data['context_pos_tags'][0])
# print(data['context_ner_tags'][0])
# print(data['context_freq_tags'][0])
# print(data['context_em_tags'][0])
# print(data['context_idxs'][0].shape)
# print(data['context_pos_tags'][0].shape)
# print(data['context_ner_tags'][0].shape)
# print(data['context_freq_tags'][0].shape)
# print(data['context_em_tags'][0].shape)
for i in range(len(data)):
    maskC = np.zeros_like(data['context_idxs'][i]) != (data['context_idxs'][i])
    lengths = (data['context_idxs'][i] != 0).sum()
    #print(lengths)
    #print(maskC.sum(-1))
    maskC_pos = np.zeros_like(data['context_pos_tags'][i]) != (data['context_pos_tags'][i])
    lengths_pos = (data['context_pos_tags'][i] != 0).sum()
    #print(maskC.sum(-1))
    maskC_ner = -np.ones_like(data['context_ner_tags'][i]) != (data['context_ner_tags'][i])
    lengths_ner = (data['context_ner_tags'][i] != -1).sum()
    # print(maskC_pos.sum(-1), maskC_pos.sum(-1), lengths_pos, lengths_ner)
    maskC = np.zeros_like(data['context_freq_tags'][i]) != (data['context_freq_tags'][i])
    #print(maskC.sum(-1))
    maskC = -np.ones_like(data['context_em_tags'][i]) != (data['context_em_tags'][i])
    #print(maskC.sum(-1))
    if maskC_pos.sum(-1) != maskC_ner.sum(-1):
        print("found mismatch")
    if lengths_pos != lengths_ner:
        print("found mismatch in lengths")