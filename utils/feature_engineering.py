# -*- coding: utf-8 -*-

'''
@Time    : 2021/9/16 14:25
@Author  : Zou Hao

'''
import os.path
import sys
from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)

import pandas as pd
import numpy as np
from pymatgen.core import Composition
from pymatgen.core import composition as comp
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from utils.elec_neg_test import check_neutrality
import json
from matminer.featurizers.composition.composite import Meredig, MagpieData, ElementProperty
from matminer.featurizers.composition.element import ElementFraction




formula = ['NaCl','H2O']
abs_file_path = os.path.dirname(__file__)


def Elfrac_fea(formulas):
    '''

    :param formula:  list of composition formulas
    :return:
    '''
    feature_cal_Elfrac = ElementFraction()
    formula_comp = [comp.Composition(n) for n in formulas]
    feature = [feature_cal_Elfrac.featurize(n) for n in formula_comp]
    feature = np.array(feature)

    return feature


def Magpie_fea(formulas):
    '''

    :param formula: list of composition formulas
    :return:
    '''

    feature_cal_Meredig = ElementProperty.from_preset('magpie')

    formula_comp = [comp.Composition(n) for n in formulas]
    feature = [feature_cal_Meredig.featurize(n) for n in formula_comp]
    feature = np.array(feature)

    return feature

def Meredig_fea(formulas):
    '''

    :param formula: list of composition formulas
    :return:
    '''

    feature_cal_Meredig = Meredig()
    feature_cal_Meredig.set_n_jobs(1)
    formula_comp = [comp.Composition(n) for n in formulas]
    feature = feature_cal_Meredig.featurize_many(formula_comp)
    feature = np.array(feature)

    return feature

def ElemNet_fea(formulas):
    '''

    :param formula: list of composition formulas
    :return:
    '''

    elements_tl = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K',
                   'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                   'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                   'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                   'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
                   'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']

    feature_cal_Elfrac = ElementFraction()
    formula_comp = [comp.Composition(n) for n in formulas]
    feature = [feature_cal_Elfrac.featurize(n) for n in formula_comp]
    feature_labels_Elfrac = feature_cal_Elfrac.feature_labels()
    data_Elfrac = pd.DataFrame(feature, columns=feature_labels_Elfrac)
    feature = np.array(feature)

    return data_Elfrac[elements_tl].values

def Roost_fea(formula,y, path):
    material_id = range(len(formula))
    roost_data = pd.DataFrame(data=material_id,columns=['materials-id'])
    roost_data['composition'] = formula
    roost_data['target'] = y
    roost_data.to_csv(path,index=False)

def ATCNN_fea(formulas):
    '''

    :param formula: list of composition formulas
    :return:
    '''
    data = pd.DataFrame(formula,columns=['composition'])
    data['comp_obj'] = data['composition'].apply(lambda x : comp.Composition(x))
    elements_tl = ['H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K',
                   'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se',
                   'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
                   'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                   'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
                   'Hg', 'Tl', 'Pb', 'Bi', 'Ac','Th', 'Pa', 'U', 'Np', 'Pu']

    feature_cal_Elfrac = ElementFraction()
    formula_comp = [comp.Composition(n) for n in formulas]
    feature = [feature_cal_Elfrac.featurize(n) for n in formula_comp]
    feature_labels_Elfrac = feature_cal_Elfrac.feature_labels()
    data_Elfrac = pd.DataFrame(feature, columns=feature_labels_Elfrac)


    at_mp_feature = []
    for i in range(len(data_Elfrac)):
        at_mp_con  = np.zeros([10,10])
        for j in range(len(elements_tl)):
            m = j // 10
            n = j % 10
            at_mp_con[m][n] =  data_Elfrac[elements_tl[j]][i]
        at_mp_feature.append(at_mp_con)

    n = len(data_Elfrac)
    at_feature = np.zeros((n, 10, 10, 1))
    for i in range(n):
        at_feature[i] = at_mp_feature[i].reshape(10, 10, 1)
    at_mp_feature = at_feature

    return np.array(at_mp_feature)



def ECCNN_fea(formulas, max_elements=15):
    """
    改进的ECCNN特征提取 - 密集表示替代稀疏矩阵

    将原来的稀疏矩阵 [batch, 8, 118, 168] 转换为密集表示 [batch, max_elements, feature_dim]

    Args:
        formulas: list of composition formulas (e.g., ['H2O', 'Fe2O3'])
        max_elements: 最大元素数量，默认15（大部分化合物元素种类<10）

    Returns:
        dict containing:
            - element_ids: [n_samples, max_elements] - 元素ID (0=padding, 1-118=元素)
            - atom_counts: [n_samples, max_elements] - 原子数目（归一化）
            - electron_configs: [n_samples, max_elements, 168] - 电子构型
            - masks: [n_samples, max_elements] - padding mask (True=padding)

    Example:
        对于 H2O:
        - element_ids: [1, 8, 0, 0, ...] (H=1, O=8)
        - atom_counts: [0.667, 0.333, 0, 0, ...] (归一化后)
        - electron_configs: [H的168维, O的168维, padding, ...]
        - masks: [False, False, True, True, ...]
    """
    # 118个元素列表（按原子序数排列）
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
                'Ds', 'Rg', 'Cn']

    # 创建元素到ID的映射 (0保留给padding)
    element_to_id = {elem: idx + 1 for idx, elem in enumerate(elements)}

    # 读取电子构型数据
    electron_config_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elec_config_one_hot.csv'))
    electron_np = electron_config_df.iloc[:, 1:-1].values.astype('float32')  # [118, 168]

    n_samples = len(formulas)

    # 初始化输出数组
    element_ids = np.zeros((n_samples, max_elements), dtype=np.int64)
    atom_counts = np.zeros((n_samples, max_elements,8), dtype=np.float32)
    #atom_counts = np.zeros((n_samples, max_elements), dtype=np.float32)
    electron_configs = np.zeros((n_samples, max_elements, 137), dtype=np.float32)
    masks = np.ones((n_samples, max_elements), dtype=bool)  # True = padding

    # 处理每个化学式
    for i, formula in enumerate(formulas):
        try:
            composition = comp.Composition(formula)
            elem_dict = composition.get_el_amt_dict()

            # 计算总原子数用于归一化
            total_atoms = sum(elem_dict.values())

            # 按元素符号排序以保证一致性
            sorted_elements = sorted(elem_dict.items(), key=lambda x: element_to_id.get(x[0], 0))

            # 填充特征
            for j, (elem, count) in enumerate(sorted_elements):
                if j >= max_elements:
                    print(f"Warning: Formula {formula} has more than {max_elements} elements, truncating.")
                    break

                if elem not in element_to_id:
                    print(f"Warning: Element {elem} not found in element list, skipping.")
                    continue

                # 元素ID
                element_ids[i, j] = element_to_id[elem]

                # 原子数目（归一化）
                #atom_counts[i,j]=count/total_atoms
                
                # === 关键修改2：原子数量 → 8位二进制向量 ===
                int_count = int(round(count))  # 确保是整数（如 2.0 → 2）
                # 转为8位二进制字符串，高位补零，例如 2 -> '00000010'
                binary_str = format(int_count, '08b')
                # 转为整数列表 [0,0,0,0,0,0,1,0]
                binary_vec = [int(bit) for bit in binary_str]
                atom_counts[i, j, :] = binary_vec  # ← 写入8维向量
                
                # 电子构型
                elem_idx = elements.index(elem)
                electron_configs[i, j, :] = electron_np[elem_idx, :]

                # 标记为非padding
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


def pero_fea(formulas):
    '''

    :param formulas:
    :return:
    '''
    # pro = pd.read_csv('../data/prop_AB.csv')
    # radii = pd.read_csv('../data/ion_radii.csv')
    data_path = os.path.dirname(__file__)
    pro = pd.read_csv(os.path.join(abs_file_path, '../data/prop_AB.csv'))
    radii = pd.read_csv(os.path.join(abs_file_path, '../data/ion_radii.csv'))
    element_set = pro['comp'].values
    U = []
    Asite = []
    A_site = []
    Bsite = []
    B_site = []

    formula_1 = []
    for i in range(len(formulas)):
        form = formulas[i]
        temp = check_neutrality(form)
        if temp[0]:
            formula_1.append([formulas[i],temp[1]])

    for i in range(len(formula_1)):
        comp = Composition(formula_1[i][0])
        ox = formula_1[i][1]
        reduce_formula = comp.get_el_amt_dict()
        element_list = list(reduce_formula.keys())
        stoichs = list(
            np.array(list(reduce_formula.values())).astype(np.float32)[:, np.newaxis]
        )
        if set(element_list) & set(element_set) != set(element_list[:-1]):
            U.append(False)
            continue

        stoichs_0 = stoichs[:-1]
        if 3 in stoichs_0:
            U.append(False)
            continue

        if stoichs[-1] == 3:
            A = element_list[0]
            A_ = element_list[0]
            B = element_list[1]
            B_ = element_list[1]
            ox_A = ox[0]
            ox_A_ = ox[0]
            ox_B = ox[1]
            ox_B_ = ox[1]
        else:
            if stoichs[0] == 2:
                A = element_list[0]
                A_ = element_list[0]
                ox_A = ox[0]
                ox_A_ = ox[0]
                if stoichs[1] == 2:
                    B = element_list[1]
                    B_ = element_list[1]
                    ox_B = ox[1]
                    ox_B_ = ox[1]
                else:
                    B = element_list[1]
                    B_ = element_list[2]
                    ox_B = ox[1]
                    ox_B_ = ox[2]
            else:
                A = element_list[0]
                A_ = element_list[1]
                ox_A = ox[0]
                ox_A_ = ox[1]
                if stoichs[2] == 2:
                    B = element_list[2]
                    B_ = element_list[2]
                    ox_B = ox[2]
                    ox_B_ = ox[2]
                else:
                    B = element_list[2]
                    B_ = element_list[3]
                    ox_B = ox[2]
                    ox_B_ = ox[3]

        if A == 'O' or B == 'O' or B_ == 'O' or A_ == 'O':
            U.append(False)
            continue

        pro_A = pro[pro['comp'] == A].values.ravel()[2:]
        pro_A_ = pro[pro['comp'] == A_].values.ravel()[2:]
        pro_B = pro[pro['comp'] == B].values.ravel()[2:]
        pro_B_ = pro[pro['comp'] == B_].values.ravel()[2:]

        A_plus = (pro_A + pro_A_) / 2
        A_min = np.abs((pro_A - pro_A_) / 2)
        B_plus = (pro_B + pro_B_) / 2
        B_min = np.abs((pro_B - pro_B_) / 2)

        radii_A = radii[(radii['Ion'] == A) & (radii['Coordination'] == 'VI')]['Ionic Radius'].values[0]
        radii_A_ = radii[(radii['Ion'] == A_) & (radii['Coordination'] == 'VI')]['Ionic Radius'].values[0]
        radii_B = radii[(radii['Ion'] == B) & (radii['Coordination'] == 'VI')]['Ionic Radius'].values[0]
        radii_B_ = radii[(radii['Ion'] == B_) & (radii['Coordination'] == 'VI')]['Ionic Radius'].values[0]
        r_X = radii[(radii['Ion'] == 'O') & (radii['Coordination'] == 'VI')]['Ionic Radius'].values[0]
        r_A = (radii_A + radii_A_) / 2
        r_B = (radii_B + radii_B_) / 2

        t = (r_A + r_X) / (2 ** 0.5 ** (r_B + r_X))
        u = r_B / r_X
        u_A = np.abs((radii_A - radii_A_) / 2)
        u_B = np.abs((radii_B - radii_B_) / 2)

        feature = list(A_plus) + list(A_min) + list(B_plus) + list(B_min) + [t, u, u_A, u_B]
        U.append(feature)
        Asite.append(A)
        A_site.append(A_)
        Bsite.append(B)
        B_site.append(B_)
    return formula_1, U, Asite, A_site, Bsite, B_site


if __name__ == '__main__':

    formulas = ['FeCoSnVO6']
    a = pero_fea(formulas)

    formulas = pd.read_csv('../data/stability_database.csv')['functional group'].values

    formula_1, U, Asite, A_site, Bsite, B_site = pero_fea(formulas)
    ab_list = []
    for i in range(len(U)):
        if len(a[i]) != 28:
            ab_list.append(i)


