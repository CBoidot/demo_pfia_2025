import numpy as np
from numpy import random
from cachetools import cached
from tqdm import tqdm

class VClassifier():
    '''Cette classe vise à représenter des processus de prédiction, qu'il s'agissent d'IA ou d'humain, et par analogie la réalité que les deux tentent de prédire.
    Elle est, comme eux, simulée ici de manière aléatoire.'''
    instances = {}
    def __init__(self,name,correlates={},seed=None):
        '''*correlates* est un dictionnaire nom-poids, où le poid joue sur l'odds_ratio de la décision.'''
        if seed is None:
            seed = random.randint(1000000)
        self.seed = seed
        self.correlates=correlates
        VClassifier.instances[name] = self

    def get_pred(name,N=1):
        return VClassifier.instances[name].pred(N)

    def get_proba(self,i,N=1):
        psum = 0
        for k,v in self.correlates.items():
            r = 2*VClassifier.get_pred(k,N)[i]-1
            psum += v*r
        odd_0 = 1+max(0,-psum)
        odd_1 = 1+max(0,psum)
        return odd_1/(odd_0+odd_1)
    
    @cached(cache={})
    def pred(self,N,i=None):
        if i is not None:
            return self.pred(vdata)[i]
        res = []
        random.seed(seed=self.seed)
        for i in range(N):
            p = self.get_proba(i,N)
            res.append(random.binomial(1,p))
        return np.array(res)
    
# Modèle de "classifieur binaire", qui va nous permettre de simuler
# aussi bien la réalité terrain, que l'IA et les agents humains.
# Chacun de ces classifieurs peut être statistiquement influencé par un autre.

class VTernaryClassifier():
    '''Cette classe vise à représenter des processus de prédiction, qu'il s'agissent d'IA ou d'humain, et par analogie la réalité que les deux tentent de prédire.
    Elle est, comme eux, simulée ici de manière aléatoire.'''
    instances = {} # mémoire nécessaire pour le système d'influence.
    def __init__(self,name,correlates={},seed=None,p_neutral=None):
        '''*correlates* est un dictionnaire nom-poids, 
        où le poids de l'influence d'un avis sur l'odds_ratio de la décision.
        p_neutral représente la probabilité désirée d'indécision : 
        si l'on ne souhaite aucune indécision, laisser sur None.'''
        if seed is None:
            seed = random.randint(1000000)
        self.seed = seed
        self.correlates=correlates
        self.name = name
        self._check_integrity()
        if p_neutral is not None:
            # we turn p to odds now
            assert p_neutral<.5
            #raise Exception('Vous ne pouvez pas être aussi indécis, si ?')
            self.neutral= abs(2-1/(p_neutral))-1 # le 2 signifie les odds de base des 2 alternatives
        else:
            self.neutral = None
        VTernaryClassifier.instances[name] = self
    
    @cached(cache={})
    def get_pred(name,N=1):
        '''méthode de classe, qui permet de récupérer une prédiction à partir du nom du classifieur.'''
        return VTernaryClassifier.instances[name].pred(N)

    def get_proba(self,i,N=1):
        '''output:
        proba_0, proba_1, proba_neutral'''
        psum = 0
        self._check_integrity()
        for k,v in self.correlates.items():
            r = 2*VTernaryClassifier.get_pred(k,N)[i]-1
            if r is not np.nan:
                psum += v*r
        odd_0 = 1+max(0,-psum)
        odd_1 = 1+max(0,psum)
        K = odd_0+odd_1
        if self.neutral is not None:
            odd_0 += self.neutral*odd_0/K
            odd_1 += self.neutral*odd_1/K
            K = odd_0+odd_1+1
        p_0, p_1 = odd_0/K, odd_1/K
        return p_0, p_1, 1-p_0-p_1
    
    @cached(cache={})
    def pred(self,N,i=None,mute=True):
        if i is not None:
            return self.pred(vdata)[i]
        res = []
        random.seed(seed=self.seed)
        gen = range(N) if mute else tqdm(range(N))
        for i in gen :
            p_0, p_1, p_nan = self.get_proba(i,N)
            dice = random.random()
            res.append(0 if dice<=p_0 else 1 if dice<=p_0+p_1 else np.nan)
        return np.array(res)
    
    def _check_integrity(self):
        if self.name in self.correlates:
            raise Exception('On ne se correle pas soi-même !')

    def _clean_instances():
        VTernaryClassifier.instances = {}