from .NoOOD import NoOOD
from .DEROG import DEROG
from .BaseOOD import BaseOODAlg

ood_alg_map = {
    'DEROG': DEROG,
    'NoOOD': NoOOD

}
