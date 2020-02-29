# Run results

## Folder structure

The structure of the folder is:
- `images/`:Segmentation en superpixel obtenue pour la base de test (Berkeley Segmentation Dataset)
- `loss-train/`: loss évaluée à chaque batch et chaque époque d'un calcul sur la base d'entraînement
(ex: run0_1.npy contient un numpy array avec les loss de chaque batch pour le run 0 et l'époque 1)
- `loss-validation/`: loss évaluée pour chaque époque sur la base de validation (on choisit comme poids
finaux du réseau sont de l'époque qui minimise la validation loss)
- `loss-plot.py`: script pour tracer les erreurs d'entraînement/validation


## Changing alpha and learning rate

| Alpha | Learning rate | Index | Recall | Precision | Underseg. | Underseg. (NP) | Compactness |
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|0|0.01|5|0.86|0.19|0.04|0.094|0.79|
|10E-6|0.01|10||||||
|10E-5|0.01|||||||
|10E-6|0.005|||||||
|10E-5|0.005|||||||
|10E-4|0.005|||||||
||||||||||
## Changing learning rate and d

|Learning rate| d |  Index |Recall|Precision|Underseg.|Underseg. (NP)|Compactness|
|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|0.01|7|0.01|5|0.86|0.19|0.04|0.094|0.79|
||||||||||



