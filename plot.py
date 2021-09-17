from matplotlib import pyplot as plt


def read_file(file_path):
    accuracies = list()
    losses = list()
    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            row = line.split()
            losses.append(float(row[1]))
            accuracies.append(float(row[2]))
    return losses, accuracies


train_loss, train_acc = read_file('train_RGB.log')
val_loss, val_acc = read_file('val_RGB.log')
n_epoch = len(train_loss)
epochs = [i for i in range(1, n_epoch+1)]

metric = 'loss'
plt.xlabel('epochs')
plt.ylabel(metric)
x_min_train, x_max_train = 0, n_epoch
y_min_train, y_max_train = 0, 6
plt.axis([x_min_train, x_max_train+1, y_min_train, y_max_train])
plt.plot(epochs, train_loss, label='training')
plt.plot(epochs, val_loss, label='validation')
plt.legend(loc='upper left')
plt.savefig(fname='{}.png'.format(metric))
plt.show()

metric = 'accuracy'
plt.xlabel('epochs')
plt.ylabel(metric)
x_min_train, x_max_train = 0, n_epoch
y_min_train, y_max_train = 0, 100
plt.axis([x_min_train, x_max_train+1, y_min_train, y_max_train])
plt.plot(epochs, train_acc, label='training')
plt.plot(epochs, val_acc, label='validation')
plt.legend(loc='upper left')
plt.savefig(fname='{}.png'.format(metric))
plt.show()