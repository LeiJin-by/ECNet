# -*- coding: utf-8 -*-


import os.path
import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)

import pandas as pd
import numpy as np
from pymatgen.core import composition as comp
import json




formula = ['NaCl','H2O']
abs_file_path = os.path.dirname(__file__)


def ECNet_fea(formulas, max_elements=15):
   

    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
                'Ds', 'Rg', 'Cn']

    element_to_id = {elem: idx + 1 for idx, elem in enumerate(elements)}


    electron_config_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elec_config_one_hot.csv'))
    electron_np = electron_config_df.iloc[:, 1:-1].values.astype('float32')  # [118, 168]

    n_samples = len(formulas)

    element_ids = np.zeros((n_samples, max_elements), dtype=np.int64)
    atom_counts = np.zeros((n_samples, max_elements,8), dtype=np.float32)
    electron_configs = np.zeros((n_samples, max_elements, 137), dtype=np.float32)
    masks = np.ones((n_samples, max_elements), dtype=bool)  # True = padding


    for i, formula in enumerate(formulas):
        try:
            composition = comp.Composition(formula)
            elem_dict = composition.get_el_amt_dict()


            total_atoms = sum(elem_dict.values())


            sorted_elements = sorted(elem_dict.items(), key=lambda x: element_to_id.get(x[0], 0))


            for j, (elem, count) in enumerate(sorted_elements):
                if j >= max_elements:
                    print(f"Warning: Formula {formula} has more than {max_elements} elements, truncating.")
                    break

                if elem not in element_to_id:
                    print(f"Warning: Element {elem} not found in element list, skipping.")
                    continue


                element_ids[i, j] = element_to_id[elem]


                

                int_count = int(round(count))  

                binary_str = format(int_count, '08b')

                binary_vec = [int(bit) for bit in binary_str]
                atom_counts[i, j, :] = binary_vec  

                elem_idx = elements.index(elem)
                electron_configs[i, j, :] = electron_np[elem_idx, :]

                masks[i, j] = False

        except Exception as e:
            print(f"Error processing formula {formula}: {e}")
            continue

    return {
        'element_ids': element_ids,
        'atom_counts': atom_counts,
        'electron_configs': electron_configs,
        'masks': masks
    }



if __name__ == '__main__':

    formulas = ['FeCoSnVO6']
   


