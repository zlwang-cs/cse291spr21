import re
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    xs = []
    train_loss = []
    valid_loss = []
    test_loss = 0
    with open('outputs/vanilla.txt') as fr:
        for line in fr:
            if 'Loss:' in line:
                proc = re.sub(r'[^0-9.]', ' ', line)
                proc = proc.strip().split()
                if 'Train' in line:
                    xs.append(len(xs) + 1)
                    loss = float(proc[0])
                    train_loss.append(loss)
                elif 'Val.' in line:
                    loss = float(proc[1])
                    valid_loss.append(loss)
                elif 'Test' in line:
                    loss = float(proc[0])
                    test_loss = loss

    plt.figure()
    l1 = plt.plot(xs, train_loss, label='train')
    l2 = plt.plot(xs, valid_loss, label='valid')
    l3 = plt.plot(xs, [test_loss for _ in range(len(xs))], label='test')
    plt.title('The loss curve of train, valid, test')
    plt.xlabel('epoch index')
    plt.ylabel('loss')
    x_ticks = np.arange(0, 15, 1)
    y_ticks = np.arange(0, 5.5, 0.5)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.legend()

    plt.show()

    print('ALL DONE')



