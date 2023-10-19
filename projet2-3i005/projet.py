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

def P2D_l(df, attr):
    # Créez un dictionnaire pour stocker les probabilités conditionnelles
    resultat = {}

    # Groupez le DataFrame par les valeurs de "target"
    mondes = df.groupby('target')

    # Parcourir chaque valeur de "target"
    for val_target, group in mondes:
        # Comptez le nombre de fois où chaque valeur d'attribut apparaît dans le groupe
        attr_counts = group[attr].value_counts().to_dict()

        # Comptez le nombre total d'occurrences de "attr" pour ce groupe
        total_count = group[attr].count()

        # Calculez les probabilités conditionnelles pour chaque valeur unique de "attr"
        prob_cond = {a: count / total_count for a, count in attr_counts.items()}

        # Ajoutez les probabilités conditionnelles au dictionnaire
        resultat[val_target] = prob_cond

    return resultat

def P2D_p(df, attr):
    resultat = {}

    # Groupez le DataFrame par les valeurs de l'attribut
    mondes = df.groupby(attr)

    # Parcourir chaque valeur de l'attribut
    for val_attr, group in mondes:
        # Comptez le nombre de fois où chaque valeur de target apparaît dans le groupe
        attr_counts = group['target'].value_counts().to_dict()

        # Comptez le nombre total d'occurrences d'une valeur de target pour ce groupe
        total_count = group['target'].count()

        # Calculez les probabilités conditionnelles pour chaque valeur de target
        prob_cond = {a: count / total_count for a, count in attr_counts.items()}

        # Ajoutez les probabilités conditionnelles au dictionnaire
        resultat[val_attr] = prob_cond

    return resultat
