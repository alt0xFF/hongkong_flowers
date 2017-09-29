import numpy as np
import matplotlib.pyplot as plt


def plot_results(log_dir):
    """

    :param log_dir: the directory name of the csv file
    :return:
    """
    data = np.genfromtxt(log_dir + 'log.csv', delimiter=',', skip_header=1)
    [epoch, train_acc, train_loss, valid_acc, valid_loss] = np.hsplit(data, 5)

    title_name = log_dir.split('/')[-2]

    param = title_name.split('_')[-1]

    # plot train functions
    plt.plot(epoch, train_loss, label='train loss: param =  %s' % param)
    plt.plot(epoch, valid_loss, label='val loss: param = %s' % param)
    plt.axis([0, len(epoch), 0, np.max((train_loss, valid_loss))])
    plt.title(title_name + ' Loss')
    plt.legend()
    plt.show()

    # plot acc functions
    plt.plot(epoch, train_acc, label='train acc: param =  %s' % param)
    plt.plot(epoch, valid_acc, label='val acc: param = %s' % param)
    plt.axis([0, len(epoch), 0, 1])
    plt.title(title_name + ' Accuracy')
    plt.legend()
    plt.show()

    best_valid_loss = np.min(valid_loss)
    best_valid_acc = np.max(valid_acc)
    best_train_loss = np.min(train_loss)
    best_train_acc = np.max(train_acc)

    print('best valid loss: %.4f, accuracy: %.4f at epoch %i' % (best_valid_loss, best_valid_acc, list(valid_loss).index(best_valid_loss)))
    print('best train loss: %.4f, accuracy: %.4f at epoch %i' % (best_train_loss, best_train_acc, list(train_loss).index(best_train_loss)))

    train_batch_loss = np.load(log_dir + 'losses.npy')
    num_batch = np.arange(0, len(train_batch_loss))

    # plot functions
    plt.plot(num_batch, train_batch_loss, label='train batch loss')
    plt.axis([0, len(num_batch), 0, np.max(train_batch_loss)])
    plt.title(title_name)
    plt.legend()
    plt.show()
