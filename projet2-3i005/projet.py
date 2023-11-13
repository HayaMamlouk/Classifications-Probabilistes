# Haya MAMLOUK [21107689]
# Maeva RAMAHATAFANDRY [21104443]

import pandas as pd
from utils import *
from scipy.stats import chi2_contingency
from scipy.stats import chi2 

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
        self.Proba_cond = P2D_l(df, attr)

    def estimClass(self, attrs):
        val_attr = attrs[self.attr]  #la valeur de l'attribut étudié du patient 
        target_0 = self.Proba_cond[0][val_attr] #P(attr | traget = 0)
        target_1 = self.Proba_cond[1][val_attr] #P(attr | traget = 1)
        
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
# completement independants : que des sommets, pas d'arcs
# sans independances : chaque sommet est relié a tous les autres sommets
#
#
#####


#####
# Question 4.2: naive Bayes
#####
# décomposition de la vraisemblance P(attr2, attr2, attr3, ... | target)
#
# P(attr1, attr2, attr3∣target) = P(attr1∣target)×P(attr2∣target)×P(attr3∣target) 
# 
#
#
#
# décomposition de la distribution a posteriori P(target | attr1, attr2, attr3, ....)
# La distribution a posteriori  P(target} |attr1, attr2, attr3, ...) est décomposée en utilisant le théorème de Bayes. La formule de Bayes est la suivante :
#
# P(target |attrs) =  ( P(attrs|target) x P(target) ) / P (attrs)
#
# Dans le contexte de la classification naïve de Bayes, on peut simplifier cette formule en utilisant l'hypothèse naïve d'indépendance conditionnelle des attributs :
#                                  n
# P(target |attrs)  ∝ P(target) x ∏i=1 P(attri|target)
# P(target |attrs)  = 1/z x P(target) x ∏i=1 P(attri|target) a
# avec Z=∑target P(target)⋅ ∏i P(attri∣target)
# Cela signifie que la distribution a posteriori est proportionnelle au produit de la probabilité a priori de la classe \( P(\text{target}) \) et du produit des probabilités conditionnelles de chaque attribut étant donné la classe.
# #####


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
        liste contenant les colonnes d'attributs que l'on veut examiner
    
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

class MLNaiveBayesClassifier(APrioriClassifier) :
    def __init__(self, df) :
        super().__init__()
        self.df = df
        self.p2dl = {} 
        for attr in df.columns:
            if(attr != 'target') :
                self.p2dl[attr]= P2D_l(self.df, attr)
        

    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance en utilisant l'hypothese du naive Bayes

        :param attrs: Un dictionnaire contenant les attributs d'un patient
        :return: Un dictionnaire contenant les valeurs de la vraisemblance
        """
        probas = {0: 1, 1: 1.0}

        for cle, val in attrs.items():
            if(cle != 'target') :
                proba_cond = self.p2dl[cle]

                target_0 = proba_cond[0].get(val, 0)  # P(attr | target = 0)
                target_1 = proba_cond[1].get(val, 0)  # P(attr | target = 1)

                probas[0] *= target_0   #P(attr1 | target == 0 ) x P(attr2 | target == 0) ...
                probas[1] *= target_1   #P(attr1 | target == 1) x P(attr2 | target == 1) ...
        return probas


    def estimClass(self, attrs):
        """
        Choisi la classe de target avec la probabilité la plus grande en utilisant le maximum de vraisemblance
        :param attrs: Un dictionnaire contenant les attributs d'un patient
        :return: la classe de target estimée
        """
        probas = self.estimProbas(attrs)
        return 0 if probas[0] >= probas[1] else 1

    
class MAPNaiveBayesClassifier(APrioriClassifier) :
    def __init__(self, df) :
        super().__init__()
        self.df = df
        self.p2dl = {}
        for attr in df.columns:
            if(attr != 'target') :
                self.p2dl[attr]= P2D_l(self.df, attr)
        
        

    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance en utilisant l'hypothese du naive Bayes

        :param attrs: Un dictionnaire contenant les attributs d'un patient
        :return: Un dictionnaire contenant les valeurs de la vraisemblance
        """
        p_target = self.df['target'].value_counts(normalize=True).to_dict()
        probas = {0: p_target[0], 1: p_target[1]}

        for cle, val in attrs.items():
            if(cle != 'target') :
                proba_cond = self.p2dl[cle]

                target_0 = proba_cond[0].get(val, 0)  # P(attr | target = 0)
                target_1 = proba_cond[1].get(val, 0)  # P(attr | target = 1)

                probas[0] *= target_0   #P(attr1 | target == 0 ) x P(attr2 | target == 0) ...
                probas[1] *= target_1   #P(attr1 | target == 1) x P(attr2 | target == 1) ...

            z = probas[0] + probas[1]
            probas[0] *= 1/z if z != 0 else 1
            probas[1] *= 1/z if z != 0 else 1
        return probas


    def estimClass(self, attrs):
        """
        Choisi la classe de target avec la probabilité la plus grande utilisant le maximum a posteriori
        :param attrs: Un dictionnaire contenant les attributs d'un patient
        :return: la classe de target estimée
        """
        probas = self.estimProbas(attrs)
        return 0 if probas[0] >= probas[1] else 1
    
def isIndepFromTarget(df, attr, x):
    tab_contingence = pd.crosstab(df[attr], df['target']) #table de contigence contenant les valeurs
    _, p, _, _ = chi2_contingency(tab_contingence) # Effectuer le test d'indépendance
    if p < x:
        return False
    else:
        return True

class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    def __init__(self, df, x):
        self.attr_indep = self.attrs_minimisees(df, x)
        self.df = df[self.attr_indep]
        super().__init__(self.df)
        
    def attrs_minimisees(self, df, x):
        attr_indep = []
        for attr in df.keys():
            if attr == 'target' or not isIndepFromTarget(df, attr, x):  #on supprime les noeuds indépendants de target
                attr_indep.append(attr)
        return attr_indep
    
    def estimClass(self, attrs):
        new_attrs = {}
        for cle, val in attrs.items():
            if cle in self.attr_indep and cle != 'target':
                new_attrs[cle] = val
        return super().estimClass(new_attrs)
    
    def estimProbas(self, attrs):
        new_attrs = {}
        for cle, val in attrs.items():
            if cle in self.attr_indep and cle != 'target':
                new_attrs[cle] = val
        return super().estimProbas(new_attrs)
    
    def draw(self):
        res = ''
        attrs = self.attr_indep.copy()
        for attr in attrs:
            if attr != "target":
                res += "{}->{};".format('target', attr)
        res = res[:-1]
        return drawGraph(res)
    
class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    def __init__(self, df, x):
        self.attr_indep = self.attrs_minimisees(df, x)
        self.df = df[self.attr_indep]
        super().__init__(self.df)
        
    def attrs_minimisees(self, df, x):
        attr_indep = []
        for attr in df.keys():
            if attr == 'target' or not isIndepFromTarget(df, attr, x):  #on supprime les noeuds indépendants de target
                attr_indep.append(attr)
        return attr_indep
    
    def estimClass(self, attrs):
        new_attrs = {}
        for cle, val in attrs.items():
            if cle in self.attr_indep and cle != 'target':
                new_attrs[cle] = val
        return super().estimClass(new_attrs)
    
    def estimProbas(self, attrs):
        new_attrs = {}
        for cle, val in attrs.items():
            if cle in self.attr_indep and cle != 'target':
                new_attrs[cle] = val
        return super().estimProbas(new_attrs)
    
    def draw(self):
        res = ''
        attrs = self.attr_indep.copy()
        for attr in attrs:
            if attr != "target":
                res += "{}->{};".format('target', attr)
        res = res[:-1]
        return drawGraph(res)
    
#####
# Question 6.1: 
#####
# Chaque point dans le graphique (précision, rappel) représente un classifieur, où l'abscisse (précision) correspond au taux de vrais positifs 
# parmi les vrais positifs et les faux positifs (atteignant sa valeur maximale de 1 lorsque le nombre de faux positifs est nul), 
# et l'ordonnée (rappel) correspond au taux de vrais positifs parmi les vrais positifs et les faux négatifs (atteignant sa valeur maximale 
# de 1 lorsque le nombre de faux négatifs est nul). Le point idéal se situe dans le coin supérieur droit du graphique, indiquant une
# précision et un rappel maximaux, signifiant l'absence totale de faux positifs et de faux négatifs. En comparant les différents classifieurs,
# nous évaluons leur performance en examinant la proximité de chaque point au coin supérieur droit, où une proximité accrue indique 
# une meilleure performance en termes de précision et de rappel.
#####

def mapClassifiers(dic, df) :
    """
    représente graphiquement les classifiers dans l'espace (précision,rappel).
    :param dic: un dictionnaire de {nom:instance de classifier}
    :param df: un dataframe
    """
    plt.figure(figsize=(6,6)) # Inirialisation du graphe

    for key, val in dic.items() :
        stats = val.statsOnDF(df)
        precision = stats['Précision']
        rappel = stats['Rappel']
        plt.scatter(precision, rappel , marker='x', color='red')
        plt.text(precision + 0.0015, rappel + 0.0015, key, fontsize=8, ha='left', va='bottom')
    plt.show

#####
# Question 6.1: 
#####
# Pour la dataframe train, le classifieur MAPNaiveBayes, utilisant des estimations a priori avec l'hypothèse du naive Bayes, 
# semble être le plus performant, étant le plus proche du coin supérieur droit sur le graphique (précision, rappel). 
# Le classifieur ReducedMAPNaiveBayes est également compétitif en termes de performances.
# Cependant, en analysant la dataframe de test, aucun classifieur ne semble être proche du coin supérieur droit. 
# Les classifieurs MLNaiveBayes et ReducedMLNaiveBayes montrent une constance dans la performance, avec une précision élevée indiquant 
# peu de faux positifs. Cependant, le rappel reste relativement bas, suggérant la présence de faux négatifs. 
# Malgré cela, les classifieurs MAPNaiveBayes et ReducedMAPNaiveBayes demeurent plus performants. Bien que le taux de faux positifs soit 
# légèrement plus élevé, le rappel est également plus élevé, démontrant une meilleure capacité à identifier les vrais positifs."
#####