import matplotlib.pyplot as plt
import pickle


class PlotHelper:
    def __init__(self, title):
        self.title = title

        self.data = []

    def add_data_point(self, dp: dict):
        self.data.append(dp)

    def plot(self):
        pass

    def save_data(self, path):
        pickle.dump({
            'title': self.title,
            'data': self.data
        }, open(path, 'wb'))


if __name__ == '__main__':
    plot_file = './outputs/train_loss_per_200_batches.pkl'
    pickle_file = pickle.load(open(plot_file, 'rb'))
    title = pickle_file['title']
    data = pickle_file['data']
