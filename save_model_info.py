import matplotlib.pyplot as plt
import os


def save_summary(string, save_path):
    with open(os.path.join(save_path, "Summary.txt"), "w") as f:
        f.write(string)


def plot_loss(history, path_to_save):
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    x = range(1, len(training_loss) + 1)

    plt.figure()
    plt.grid(True)

    plt.plot(x, training_loss, color="blue", label="Training loss")
    plt.plot(x, validation_loss, color="red", label="Validation loss")
    
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(os.path.join(path_to_save, "Loss.png"), bbox_inches="tight")



def plot_history(history, path_to_save):
    plot_loss(history, path_to_save)
