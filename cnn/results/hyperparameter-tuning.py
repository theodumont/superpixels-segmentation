"""
Hyperparameter tuning
"""
import numpy as np
import matplotlib.pyplot as plt

## d

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
plt.savefig("graphs/hpp-d.png")
# plt.show()
print([d4, d5, d6, d7])

plt.figure("Hyperparameter loss")
plt.plot(validation_loss4, label="d=4")
plt.plot(validation_loss5, label="d=5")
plt.plot(validation_loss6, label="d=6")
plt.plot(validation_loss7, label="d=7")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.savefig("graphs/hpp-d-loss.png")
# plt.show()

