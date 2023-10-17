# Haya MAMLOUK [21107689]
# Maeva RAMAHATAFANDRY [21104443]

import pandas as pd

def getPrior(classe) :
    data = pd.read_csv(classe)
    nb_patient = len(data) - 1 
    target = data[data["target"] == 1]
    nb_malades = target.count()
    estimation = target / nb_patient
    return estimation


