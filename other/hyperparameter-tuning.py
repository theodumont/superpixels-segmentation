"""
Hyperparameter tuning
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
sb.set()
# d

if False:
    # consider the runs 3, 4, 5 and 0
    validation_loss4 = np.load("loss-validation/Validation_loss_3.npy")
    validation_loss5 = np.load("loss-validation/Validation_loss_4.npy")
    validation_loss6 = np.load("loss-validation/Validation_loss_5.npy")
    validation_loss7 = np.load("loss-validation/Validation_loss_0.npy")
    d4 = validation_loss4[25]
    d5 = validation_loss5[29]
    d6 = validation_loss6[34]
    d7 = validation_loss7[36]

    plt.figure("Minimum loss")
    # figsize=(3,2.5)
    plt.plot([4, 5, 6, 7], [d4, d5, d6, d7])
    plt.xlabel("d")
    plt.ylabel("Loss")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # plt.savefig("graphs/hpp-d.png")
    # plt.show()
    print([d4, d5, d6, d7])

    plt.figure("Hyperparameter loss")
    plt.plot(validation_loss4, label="d = 4")
    plt.plot(validation_loss5, label="d = 5")
    plt.plot(validation_loss6, label="d = 6")
    plt.plot(validation_loss7, label="d = 7")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.legend()
    plt.savefig("graphs/hpp-d-loss.png")
    plt.show()

# lr

if False:
    # consider the runs 6 and 8
    # validation_loss6 = np.load("loss-validation/Validation_loss_6.npy")
    validation_loss7 = np.load("loss-validation/Validation_loss_7.npy")
    # validation_loss8 = np.load("loss-validation/Validation_loss_8.npy")
    validation_loss9 = np.load("loss-validation/Validation_loss_9.npy")
    validation_loss10 = np.load("loss-validation/Validation_loss_10.npy")
    plt.figure("Loss")
    # plt.plot(validation_loss6, label='(run 6) .01, /2')
    plt.plot(validation_loss7, label='(run 7) .001')
    # plt.plot(validation_loss8, label='(run 8) .01')
    plt.plot(validation_loss9, label='(run 9) .001, /2')
    plt.plot(validation_loss10, label='(run 10) .001')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("graphs/hpp-lr-loss-7910.png")
    # plt.show()


# alpha TV

if True:
    validation_loss9 = np.load("loss-validation/Validation_loss_9.npy")
    # validation_loss12 = np.load("loss-validation/Validation_loss_12.npy")
    validation_loss13 = np.load("loss-validation/Validation_loss_13.npy")
    plt.figure("Loss")
    plt.plot(validation_loss9, label='(run 9) alpha=0')
    plt.plot(validation_loss13, label='(run 13) alpha=1E-7')
    # plt.plot(validation_loss12, label='(run 12) alpha=1E-6')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("graphs/hpp-tv-loss.png")
    plt.figure("Minimum loss")
    # figsize=(3,2.5)
    plt.plot([4, 5, 6, 7], [d4, d5, d6, d7])
    plt.xlabel("d")
    plt.ylabel("Loss")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig("graphs/hpp-tv.png")
    plt.show()
    print([d4, d5, d6, d7])
