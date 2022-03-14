import matplotlib
import seaborn as sns
import numpy as np
# import spacy
import ujson as json
from ujson import load as json_load

matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # drawing heat map of attention weights

#plt.rcParams['font.sans-serif'] = ['SimSun']  # set font family
# nlp = spacy.blank("en")

with open('data/idxtoword.json', 'r') as fh:
    gold_dict = json_load(fh)

def idx2word(dict, w_idxs):
    return [dict[str(idx)] for idx in w_idxs[1:]]


def plot_attention(data, file_name, cw_idxs, qw_idxs):
    '''
      Plot the attention model heatmap
      Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        X_label: list of size tx, encoder tags
        Y_label: list of size ty, decoder tags
    '''
    X_label = idx2word(gold_dict, cw_idxs.tolist())
    #print(Y_label)
    Y_label = idx2word(gold_dict, qw_idxs.tolist())
    #print(len(X_label))
    #print(data.shape)
    fig, ax = plt.subplots(figsize=(30, 12))  # set figure size
    #plt.imshow(data, cmap='hot', interpolation='nearest')
    ax = sns.heatmap(data, cmap='hot')
    #ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
    # Set axis labels
    if X_label != None and Y_label != None:
        #X_label = [x_label for x_label in X_label]
        #Y_label = [y_label for y_label in Y_label]

        xticks = range(0, len(X_label))
        ax.set_xticks(xticks, minor=False)  # major ticks
        ax.set_xticklabels(X_label, minor=False, rotation=45)  # labels should be 'unicode'

        yticks = range(0, len(Y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(Y_label, minor=False)  # labels should be 'unicode'

        ax.grid(True)

    # Save Figure
    plt.title(u'Attention Heatmap')
    print("Saving figures %s" % file_name)
    fig.savefig(file_name)  # save the figure to file
    plt.close(fig)  # close the figure
