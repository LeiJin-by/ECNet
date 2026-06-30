# -*- coding: utf-8 -*-

import json
import os
import sys
from os.path import abspath, dirname

import numpy as np
import pandas as pd
from pymatgen.core import composition as comp


path = dirname(dirname(abspath(__file__)))
sys.path.append(path)


ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
    'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
    'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
    'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
    'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
    'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
    'Ds', 'Rg', 'Cn'
]


def parse_oxidation_states(value):
    if isinstance(value, dict):
        states = value
    else:
        states = json.loads(value)
    return {str(element): int(state) for element, state in states.items()}


def load_oxidation_lookup(config_path=None):
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'elec_config_oxidation_state_all.csv',
        )
        if not os.path.exists(config_path):
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'elec_config_oxidation_state.csv',
            )
    table = pd.read_csv(config_path)
    config_cols = [f'column_{i}' for i in range(1, 138)]
    lookup = {}
    for _, row in table.iterrows():
        key = (row['symbol'], int(row['oxidation_state']))
        lookup[key] = row[config_cols].values.astype('float32')
    return lookup


def ECNet_ox_fea(formulas, oxidation_states, max_elements=15, config_path=None):
    element_to_id = {elem: idx + 1 for idx, elem in enumerate(ELEMENTS)}
    oxidation_lookup = load_oxidation_lookup(config_path)

    n_samples = len(formulas)
    element_ids = np.zeros((n_samples, max_elements), dtype=np.int64)
    atom_counts = np.zeros((n_samples, max_elements, 8), dtype=np.float32)
    electron_configs = np.zeros((n_samples, max_elements, 137), dtype=np.float32)
    masks = np.ones((n_samples, max_elements), dtype=bool)

    for i, formula in enumerate(formulas):
        try:
            composition = comp.Composition(formula).reduced_composition
            elem_dict = composition.get_el_amt_dict()
            ox_states = parse_oxidation_states(oxidation_states[i])
            sorted_elements = sorted(elem_dict.items(), key=lambda x: element_to_id.get(x[0], 0))

            for j, (elem, count) in enumerate(sorted_elements):
                if j >= max_elements:
                    print(f"Warning: Formula {formula} has more than {max_elements} elements, truncating.")
                    break

                if elem not in element_to_id:
                    print(f"Warning: Element {elem} not found in element list, skipping.")
                    continue

                if elem not in ox_states:
                    raise KeyError(f"Missing oxidation state for {elem} in {formula}")

                ox_state = int(ox_states[elem])
                lookup_key = (elem, ox_state)
                if lookup_key not in oxidation_lookup:
                    raise KeyError(f"Missing oxidation-state vector for {elem}{ox_state:+d}")

                element_ids[i, j] = element_to_id[elem]
                int_count = int(round(count))
                binary_str = format(int_count, '08b')
                atom_counts[i, j, :] = [int(bit) for bit in binary_str]
                electron_configs[i, j, :] = oxidation_lookup[lookup_key]
                masks[i, j] = False

        except Exception as e:
            print(f"Error processing formula {formula}: {e}")
            continue

    return {
        'element_ids': element_ids,
        'atom_counts': atom_counts,
        'electron_configs': electron_configs,
        'masks': masks,
    }
