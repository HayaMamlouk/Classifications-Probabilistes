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
        return 1
        # # global a_priori
        # a_priori = getPrior(train)

        # # Récupérez l'estimation de la probabilité a priori
        # prior_estimation = a_priori['estimation']
        
        # # Déterminez un seuil 
        # seuil = 0.5
        
        # # Si l'estimation est supérieure au seuil, le patient est prédit comme malade (1), sinon non malade (0)
        # if prior_estimation > seuil:
        #     return 1
        # return 0

    # def estimClass(self, attrs):
    #     """
    #     L'idée est de trouver tous les patients avec un même profil que attrs, calculer la proba qu'il soit malade, et vérifier s'il se trouve dans l'intervalle de confiance.  Demander au prof
    #     On va travailler avec la classe train pour le moment                                                 
    #     """
    #     # DEMANDE AU PROF
    #     data = train 

    #     # Supprimez la colonne "target" du DataFrame
    #     data_WO_target = data.drop("target", axis=1)

    #     # Supprimez également la clé "target" du dictionnaire d'attributs
    #     attrs_WO_target = {key: value for key, value in attrs.items() if key != "target"}

    #     # Trouvez les profils identiques 
    #     profil_identiques = data[data_WO_target.eq(attrs_WO_target).all(axis=1)]

    #     # Calculez le nombre de profils identiques et le nombre de malades parmi eux
    #     len_ident = len(profil_identiques)
    #     len_malade = len(profil_identiques[profil_identiques['target'] == 1])

    #     # Estime la probabilité
    #     estim_attrs = len_malade / len_ident

    #     # Obtenez l'estimation a priori
    #     apriori = getPrior(train)

    #     if apriori is not None:
    #         # Obtenez les bornes de l'intervalle de confiance
    #         min5pourcent = apriori['min5pourcent']
    #         max5pourcent = apriori['max5pourcent']

    #         # Vérifiez si l'estimation est en dehors de l'intervalle de confiance
    #         if estim_attrs >= min5pourcent and estim_attrs <= max5pourcent:
    #             return 1

    #     return 0
  

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

            