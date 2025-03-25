import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import numpy as np
import scipy
import math
import pickle
import json
from utils import gauss


def plotAccLoss(opt,
                train_losses,
                val_losses,
                train_accs,
                val_accs, 
                yscale='linear',
                save_path=None,
                extra_pt=None,
                extra_pt_label=None):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    epoch_array = np.arange(len(train_losses)) + 1
    sns.set(style='ticks')

    legends = ['Train', 'Validation']
    ax1.plot(epoch_array, train_losses, epoch_array, val_losses, linestyle='dashed', marker='o')
    ax1.legend(legends)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training')
    ax1.set_title('Loss')
    
    ax2.plot(epoch_array, train_accs, epoch_array, val_accs, linestyle='dashed', marker='*')
    ax2.legend(legends)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation')
    ax2.set_title('Accuracy')

    sns.despine(trim=True, offset=5)
    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight')
    fig.tight_layout()
    
    plt.savefig(f"../../results/{opt.phase}/{opt.name}.png")
    json_data = {
        "train loss": train_losses,
        "validation loss": val_losses,
        "train accuracy": train_accs,
        "validation accuracy": val_accs
    }
    json_data = json.dumps(json_data, indent=4)
    with open("../../results/"+opt.phase+"/"+opt.name+".json", "w") as jsonfile:
        jsonfile.write(json_data)



def plot_data_length_stats(texts, opt):
    mydata = [list(texts[i].values())[0][0] for i in range(len(texts))]
    data = [len(mydata[i].split(" ")) for i in range(len(mydata))]
    mydataNr = [i for i in range(len(data))]
    figure, axis = plt.subplots(2, 2)

    axis[0, 0].scatter(mydataNr, data, color="violet", marker='.')
    axis[0, 0].set_title("Conversation words count of all data instance", size=24)
    axis[0, 0].set_xlabel("Index number of all data", fontsize=22)
    axis[0, 0].set_ylabel("Conversation words count number of each data instance", fontsize=20)

    n, bins, patches = axis[0, 1].hist(data, bins=300, color='skyblue')
    x = np.arange(len(n))
    popt, pcov = curve_fit(gauss, x, n, maxfev=50000)
    x = np.arange(len(bins))
    y = [gauss(a, popt[0], popt[1], popt[2], popt[3]) for a in x]
    axis[0, 1].plot(bins, y, color="tomato")
    axis[0, 1].set_title(f"Histogram of words number with {len(bins)-1} bin and gaussian fitting", size=24)
    axis[0, 1].set_xlabel("300 bins for words count of conversations", fontsize=22)
    axis[0, 1].set_ylabel("Instances count of each bin", fontsize=20)

    H = popt[0]
    A = popt[1]
    mu = popt[2]
    sigma = popt[3]
    confidence_level = opt.words_count_cover_confidence_level
    h = sigma * scipy.stats.t.ppf((1 + confidence_level) / 2., len(x) - 1)
    rightside = math.ceil(mu+h)
    height = gauss(rightside, H, A, mu, sigma)
    x = np.arange(rightside+1)
    y = [gauss(a, H, A, mu, sigma) for a in x]
    axis[1, 0].hist(data, bins=300, color='skyblue')
    axis[1, 0].plot(bins[:rightside+1], y, color="tomato")
    axis[1, 0].fill_between(bins[:rightside+1], y, interpolate=True, color='tomato')
    axis[1, 0].scatter(bins[rightside], height, color="blue", marker='^')
    axis[1, 0].text(bins[rightside], int(height), f"   ({int(bins[rightside])},  {int(height)})")
    axis[1, 0].set_title(f"Confidence interval of confidence level = {confidence_level}", size=24)
    axis[1, 0].set_xlabel("300 bins for words count of conversations", fontsize=22)
    axis[1, 0].set_ylabel("Instances count of each bin", fontsize=20)

    
    axis[1, 1].hist(data, bins=300, color='skyblue')
    axis[1, 1].plot(bins[:rightside+1], y, color="tomato")
    axis[1, 1].fill_between(bins[:rightside+1], y, interpolate=True, color='tomato')
    axis[1, 1].scatter(bins[rightside], height, color="blue", marker='^')
    axis[1, 1].text(bins[rightside], int(height), f"   ({int(bins[rightside])},  {int(height)})")
    axis[1, 1].set_title("Zoom in of confidence interval", size=24)
    axis[1, 1].set_xlabel("300 bins for words count of conversations", fontsize=22)
    axis[1, 1].set_ylabel("Instances count of each bin", fontsize=20)


    figure.tight_layout()
    plot_file = f"../../results/{opt.phase}/Cut{opt.extreme_conversation_cut}_Words_length_states.pickle"
    pickle.dump(figure, open(plot_file, 'wb'))
    print(f"Cutoff of extreme long conversation: {opt.extreme_conversation_cut}, {opt.words_count_cover_confidence_level} confidence level of the cover of words count of conversations is  {int(bins[rightside])}")
    
    return int(bins[rightside])
