import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

if __name__ == '__main__':
    file_path = './outputs/log_temp.txt'
    pattern_a = re.compile(r'\| epoch\s*(.*) \|\s*(.*)/ 2983 batches \| lr\s*(.*) \| ms/batch\s*(.*) \| loss\s*(.*) \| ppl\s*(.*)')
    pattern_b = re.compile(r'\| end of epoch\s*(.*) \| time:\s*(.*)s \| valid loss\s*(.*) \| valid ppl\s*(.*)')

    n_batch_per_epoch = 2983

    train_loss_per_200_epochs = {'xs': [], 'train': []}
    train_valid_loss_per_epoch = {'xs': [], 'train': [], 'valid': []}

    train_loss_list = []

    with open(file_path) as fr:
        for line in fr:
            line = line.strip()
            match_a = pattern_a.match(line)
            match_b = pattern_b.match(line)

            if match_a and match_a.group():
                epoch = int(match_a.group(1))
                batch = int(match_a.group(2))
                train_loss = float(match_a.group(5))

                train_loss_per_200_epochs['xs'].append(n_batch_per_epoch * (epoch - 1) + batch)
                train_loss_per_200_epochs['train'].append(train_loss)

                train_loss_list.append(train_loss)

            elif match_b and match_b.group():
                epoch = int(match_b.group(1))
                valid_loss = float(match_b.group(3))

                train_valid_loss_per_epoch['xs'].append(epoch)
                train_valid_loss_per_epoch['valid'].append(valid_loss)
                train_valid_loss_per_epoch['train'].append(sum(train_loss_list)/len(train_loss_list))

                train_loss_list.clear()

    plt.figure(figsize=(5, 10))
    ax1 = plt.subplot(2, 1, 1,)
    plt.plot(train_loss_per_200_epochs['xs'], train_loss_per_200_epochs['train'])
    plt.title('The training loss per 200 batches')
    plt.xlabel('batch index')
    plt.ylabel('training loss')
    y_ticks = np.arange(3.5, 8, 0.5)  # 取值(-2,2)范围，采样11个点
    plt.yticks(y_ticks)

    ax2 = plt.subplot(2, 1, 2)
    l1 = plt.plot(train_valid_loss_per_epoch['xs'], train_valid_loss_per_epoch['train'], label='train')
    l2 = plt.plot(train_valid_loss_per_epoch['xs'], train_valid_loss_per_epoch['valid'], label='valid')
    plt.title('The training and the validation losses per epoch')
    plt.xlabel('epoch index')
    plt.ylabel('loss')
    x_ticks = np.arange(0, 16, 1)
    y_ticks = np.arange(3.5, 8, 0.5)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.legend()

    plt.show()

    print('ALL FINISH!')
