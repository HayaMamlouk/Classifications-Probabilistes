# Haya MAMLOUK [21107689]
# Maeva RAMAHATAFANDRY [21104443]

import pandas as pd
from utils import *

def getPrior(data):
    """
    Calcule la probabilité a priori de la classe data ainsi que l'intervalle de confiance à 95% pour l'estimation de cette probabilité.
    """
    # NB : Les données sont déja chargé dans le ipnyb

    # Calculer le nombre total de patients
    nb_patient = len(data)

    # Filtrer les lignes où la colonne "target" est égale à 1 (malades)
    target = data[data["target"] == 1]

    # Calculer le nombre de malades
    nb_malades = len(target)

    # Calculer l'estimation des malades
    estimation = nb_malades / nb_patient

    # Calculer l'intervalle de confiance à 95% pour l'estimation
    conf_int = (estimation - 1.96 * (estimation * (1 - estimation) / nb_patient) ** 0.5,
                estimation + 1.96 * (estimation * (1 - estimation) / nb_patient) ** 0.5)

    # Créer un dictionnaire contenant les résultats
    result = {
        'estimation': estimation,
        'min5pourcent': conf_int[0],
        'max5pourcent': conf_int[1]
    }

    return result



class APrioriClassifier(AbstractClassifier) :
    def __init__(self) :
        super().__init__()

    def estimClass(self, attrs):
        """Renvoie la classe majoritaire (malade == 1)"""
        return 1

    def statsOnDF(self, df):
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.
        """
        VP, VN, FP, FN = 0, 0, 0, 0,
        for _, row in df.iterrows() :
            attrs = row.drop('target')
            classe_prevue = self.estimClass(attrs)
            vraie_classe = row['target']

            if classe_prevue == 1 and vraie_classe == 1 :
                VP +=1
            elif classe_prevue == 0 and vraie_classe == 0 :
                VN +=1
            elif classe_prevue == 1 and vraie_classe == 0 :
                FP +=1
            else :
                FN += 1

        precision = VP / (VP + FP)
        rappel = VP / (VP + FN)

        result = {
            'VP' : VP,
            'VN' : VN,
            'FP' : FP,
            'FN' : FN,
            'Précision' : precision,
            'Rappel' : rappel
        }

        return result

            