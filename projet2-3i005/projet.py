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
    for val_node, group in mondes:
        # Comptez le nombre de fois où chaque valeur d'attribut apparaît dans le groupe
        attr_counts = group[attr].value_counts().to_dict()

        # Comptez le nombre total d'occurrences de "attr" pour ce groupe
        total_count = group[attr].count()

        # Calculez les probabilités conditionnelles pour chaque valeur unique de "attr"
        prob_cond = {a: count / total_count for a, count in attr_counts.items()}

        # Ajoutez les probabilités conditionnelles au dictionnaire
        resultat[val_node] = prob_cond

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

class ML2DClassifier(APrioriClassifier) :
    def __init__(self, df, attr) :
        super().__init__()
        self.attr = attr
        self.P2Dl = P2D_l(df, attr)

    def estimClass(self, attrs):
        val_attr = attrs[self.attr]  #la valeur de l'attribut étudié du patient 
        target_0 = self.P2Dl[0][val_attr] #P(attr | traget = 0)
        target_1 = self.P2Dl[1][val_attr] #P(attr | traget = 1)
        
        return 0 if target_0 >= target_1 else 1 #rend le target avec la probabilté la plus grde, 0 si égales
    
class MAP2DClassifier(APrioriClassifier) :
    def __init__(self, df, attr) :
        super().__init__()
        self.attr = attr
        self.P2Dp = P2D_p(df, attr)

    def estimClass(self, attrs):
        val_attr = attrs[self.attr]  #la valeur de l'attribut étudié du patient 
        target_0 = self.P2Dp[val_attr][0] #P(traget = 0 | attr )
        target_1 = self.P2Dp[val_attr][1] #P(traget = 1 | attr )
        
        return 0 if target_0 >= target_1 else 1 #rend le target avec la probabilté la plus grde, 0 si égales

def count_values(df):
    """
    Cette fonction cree un dictionnaire des éléments uniques, pour chaque attributcontenus dans la
    dataframe donnée argument tel que cle = colomne et valeur = nombre valeurs uniques

    Parameters
    ----------
    df: pd.dataframe
        dataframe contenant n colonnes dont il faut identifier les valeurs
    
    Returns
    -------
    res: dict()  
        dictionnaire contenant les valeurs uniques de chaque colonne 

    """
    res = {}
    for cle, valeur in df.items():
        temp = []
        for val in valeur:
            if val not in temp:
                temp.append(val)
        res[cle] = len(temp)
    return res

def nbParams(df, attrs=None):
    """
    Cette fonction affiche la taille mémoire de chaque colomne contenue dans df

    Parameters
    ----------
    df: pd.dataframe
        dataframe contenant les données de chaque attributs pour chaque élement
        de la population
    
    attrs: list()
        liste contenant les colomnes d'attributs que l'on veut examiner
    
    Returns
    -------
    nb_oct: nombre d'octets total pour le dataframe et les attributs correspondant
    
    """
    if attrs is not None:
        df = df[attrs]
    
    nb_oct = 1

    for value in count_values(df).values():
        nb_oct *= value
    nb_oct *= 8

    print(len(df.keys()), "variable(s) : ", nb_oct, " octets")

    return nb_oct

def nbParamsIndep(df, attrs=None):
    """
    Cette fonction affiche la taille mémoire nécessaire pour représentaer les tables de probabilités
    en supposant l'indépendance des variables

    Parameters
    ----------
     df: pd.dataframe
        dataframe contenant les données de chaque attributs pour chaque élement
        de la population
    
    attrs: list()
        liste contenant les colomnes d'attributs que l'on veut examiner
    
    Returns
    -------
    nb_oct: nombre d'octets total pour le dataframe et les attributs correspondant
    """
    if attrs is not None:
        df = df[attrs]
    
    nb_oct = 0

    for value in count_values(df).values():
        nb_oct += value
    nb_oct *= 8

    print(len(df.keys()), "variable(s) : ", nb_oct, " octets")

    return 

#####
# Question 4.1: Exemples
#####
#
#
#
#
#####


#####
# Question 4.2: naive Bayes
#####
# décomposition de la vraisemblance P(attr2, attr2, attr3, ... | target)
#
#
#
#
#
#
# décomposition de la distribution a posteriori P(target | attr1, attr2, attr3, ....)
#####

def drawNaiveBayes(df, attr):
    """
    Cette fonction dessine le graphe qui représente un modèle naive bayes 

    Parameters
    ----------
     df: pd.dataframe
        dataframe contenant les données de chaque attributs pour chaque élement
        de la population
    
    attr: str
        chaîne de caractère contenant le nom de la colonne qui est la classe
    
    Returns
    -------
     Image
      l'image représentant le graphe
    """
    str_columns = []
    res = "" #str contenant la chaine
    for cle, _ in df.items():
        if cle != attr :
            res += attr + "->" + cle + ";"
    
    return drawGraph(res[:-1])

def nbParamsNaiveBayes(df, node, attrs=None):
    """
    Cette fonction écrit la taille mémoire nécessaire pour représenter les tables 
    de probabilité etant donné un dataframe
    Parameters
    ----------
    df: pd.dataframe
        dataframe contenant les données de chaque attributs pour chaque élement
        de la population
    
    node : str
        noeud target, parent de tous les attributs
    
    attrs: list()
        liste contenant les colomnes d'attributs que l'on veut examiner
    
    Returns
    -------
    nb_oct: nombre d'octets total pour le dataframe et les attributs correspondant
    
    """
    val_node = 0
    nb_oct = 1

    if attrs is not None: 
        old_df = df
        df = df[attrs]
        if len(attrs) == 0:
            val_node = old_df[node].nunique()

    for key,value in count_values(df).items():
        if key == node:
            val_node += value
        else:
            nb_oct += value

    nb_oct *= val_node 
    nb_oct *= 8

    print(len(df.keys()), "variable(s) : ", nb_oct, " octets")

    return nb_oct

    
