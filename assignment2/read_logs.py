import re
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    log_file = './logs/CRF-final.log'

    train_loss_list = []
    valid_loss_list = []

    train_loss = None
    valid_loss = None
    with open(log_file) as fr:
        for line in fr:
            line = line.strip()
            if 'EPOCH' in line:
                # m = re.match(r'--- EPOCH ([0-9*]) ---', line)
                # cur_epoch = m.group(1)
                train_loss = -1
                valid_loss = -1
            if 'Avg evaluation loss:' in line:
                loss = float(line.rsplit(' ', 1)[1])
                if train_loss == -1:
                    train_loss = loss
                    train_loss_list.append(loss)
                elif valid_loss == -1:
                    valid_loss = loss
                    valid_loss_list.append(loss)
            if '(' == line[0] and ')' == line[-1]:
                acc, rec, f1 = eval(line)
                if best is None or f1 > best[-1]:
                    print(line)
                    best = (acc, rec, f1)

    print('BEST:', best)

    plt.figure()
    xs = list(range(len(train_loss_list)))
    l1 = plt.plot(xs, train_loss_list, label='train')
    l2 = plt.plot(xs, valid_loss_list, label='valid')
    # plt.title('The loss curve of train, valid')
    plt.xlabel('epoch index')
    plt.ylabel('loss')
    x_ticks = np.arange(0, 100, 5)
    y_ticks = np.arange(0, 7, 0.5)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.legend()

    plt.show()

    print('ALL DONE')




