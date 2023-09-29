import pandas as pd
import numpy as np
import math
import re
import torch

from PyAstronomy import pyasl

from utils.constants import *

def get_elements_and_compositions(x, verbose = -1):
    """
    Parameters
    ----------
    x : str
        The string of the alloy
        Example: "Co14Ni69P17", etc.
    
    Returns
    -------
    elements: list[str]
        all the elements in the alloy
        Example: ["Co", "Ni", "P"]
    compositions: list[float]
        all the compositions of elements in the alloy
        Example: [14.0, 69.0, 17.0]
    """
    # separating atoms from composition
    s = re.sub(r'[^\w\s]','',x)
    s = re.sub('\d',' ',s)
    elements = np.array([i for i in s.split(' ') if i != ""])
    if verbose > 0:
        print('\nElements in BMG are : ', elements)

    compositions = re.findall(r"[-+]?\d*\.\d+|\d+", x)
    compositions = [float(i) for i in compositions]
    if verbose > 0:
        print('Compositions: ', compositions)
    
    return elements, compositions

def element_to_index(element):
    """
    Parameters
    ----------
    element : str
        The element that you want the atomic number of
        Example: "Al"

    Returns
    -------
    atomic_number : int
        Example: 13
    """
    try:
        atomic_number = pyasl.AtomicNo()
        return atomic_number.getAtomicNo(element)
    except:
        return "END"

def index_to_element(index):
    """
    Parameters
    ----------
    index : int
        The atomic number that you want the element of
        Example: 13
    
    Returns
    -------
    The element with the atomic number
        Example: "Al"
    """
    try:
        atomic_number = pyasl.AtomicNo()
        return atomic_number.getElSymbol(index)
    except:
        return "END"
    
def combine_elements_and_composition_to_alloy(elements, compositions):
    """
    Parameters
    ----------
    elements : list[str]
        all the elements in the alloy
        Examples: ["Co", "Ni", "P"]
    compositions: list[float]
        all the compositions of elements in the alloy
        Example: [14.0, 69.0, 17.0]
    
    Returns
    -------
    The string of the alloy
        Example: "Co14Ni69P17", etc.
    """
    alloy = ""
    for idx in range(len(elements)):
        alloy += str(index_to_element(elements[idx])) + str(compositions[idx])[:4]
    return alloy.split("END")[0]

def combine_elements_and_composition_to_alloy_already_string(elements, compositions):
    alloy = ""
    for idx in range(len(elements)):
        alloy += str(elements[idx]) + str(compositions[idx])[:4]
    return alloy.split("END")[0]

def order_df(df, alloy_col_name="bmg_alloy"):
    """
    Rearrange the alloy strings by increasing atomic number
    This results in the same alloy just rearranged

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the alloy strings.

    Returns
    -------
    DataFrame
        The modified DataFrame with alloy strings rearranged by increasing atomic number.
    "
    """
    ordered_alloys = []
    for i in range(len(df)):
        elements, compositions = get_elements_and_compositions(df.iloc[i][alloy_col_name])
        # print(elements, compositions)
        all_element_atomic_numbers = []
        for element in elements:
            all_element_atomic_numbers.append(element_to_index(element))
        ordered_indices = np.argsort(all_element_atomic_numbers)
        ordered_elements = np.take(elements, ordered_indices)
        ordered_compositions = np.take(compositions, ordered_indices)
        print("ordered:", ordered_elements, ordered_compositions)
        ordered_alloy = combine_elements_and_composition_to_alloy_already_string(ordered_elements, ordered_compositions)
        ordered_alloys.append(ordered_alloy)
    print(ordered_alloys)
    df[alloy_col_name] = ordered_alloys
    return df

# basic utility functions to calculate features

def diff(alloy):
    """
    Compares the atoms in the given alloy and separates them into 'big' and 'small' groups based on a scoring matrix.

    Parameters
    ----------
    alloy : list
        A list of atoms in the alloy.

    Returns
    -------
    big : list
        A list of atoms considered 'big' based on the scoring matrix.
    small : list
        A list of atoms considered 'small' based on the scoring matrix.
    """
    # making ranges for each atom
    ranges = {}
    for i in alloy:
        ranges[i] = 0.88 * parameters[i]["ar"]
    # compiling scoring matrix
    score = {}
    for i in alloy:
        current_score = {}
        for j in alloy:
            if parameters[i]["ar"] < ranges[j]:
                current_score[j] = -1
            elif parameters[i]["ar"] > parameters[j]["ar"]:
                current_score[j] = 1
            else:
                current_score[j] = 0
        score[i] = current_score

    big = []
    small = []
    # separating into big and small based on scoring matrix
    for i in score:
        total_sum = 0
        for j in score[i]:
            total_sum = total_sum + score[i][j]
        if total_sum > 0:
            big.append(i)
        else:
            small.append(i)

    if len(big) == 0 or len(small) == 0:
        print(score)
    return big, small
    
# finds the paramater deltaE
def electro(elements_and_compositions, alloy):
    """
    This function calculates the average electronegativity of an alloy. 

    Parameters
    ----------
    elements_and_compositions : dict
        A dictionary where keys are the elements in the alloy and values are their respective compositions.
    alloy : list
        A list of elements that make up the alloy.
        The function works by first calculating the total composition of the alloy. Then, it calculates the sum of the product of each element's composition and its electronegativity. Finally, it returns the average electronegativity by dividing the sum of the product by the total composition.

    Returns
    -------
    The average electronegativity of the alloy.
    """
    summation_of_product_of_composition_and_electronegativity = 0 # summation of product of composition and electro negativity
    summation_of_composition = 0 # summation of compositions
    for i in elements_and_compositions:
        if i in alloy:
            summation_of_composition = summation_of_composition + elements_and_compositions[i]
            
    for i in elements_and_compositions:
        if i in alloy:
            summation_of_product_of_composition_and_electronegativity = summation_of_product_of_composition_and_electronegativity + elements_and_compositions[i] * parameters[i]['en']
    return summation_of_product_of_composition_and_electronegativity / summation_of_composition

# finds the paramater deltaD
def comps(elements_and_compositions, alloy):
    summation_of_product_of_composition_and_atomicradii = 0 # summation of product of composition and atomic radii
    summation_of_compositions = 0 # summation of compositions
    for i in elements_and_compositions:
        if i in alloy:
            summation_of_compositions = summation_of_compositions + elements_and_compositions[i]
    for i in elements_and_compositions:
        if i in alloy:
            summation_of_product_of_composition_and_atomicradii = summation_of_product_of_composition_and_atomicradii + elements_and_compositions[i] * parameters[i]['ar']
    return summation_of_product_of_composition_and_atomicradii / summation_of_compositions

def prepare_params(alloy):
    # separating atoms from composition
    s = re.sub(r'[^\w\s]','', alloy)
    s = re.sub('\d', ' ', s)
    elements = np.array([i for i in s.split(' ') if i in parameters]) # elements list
    # print('\nElements in BMG are : ', elements)

    compositions = re.findall(r"[-+]?\d*\.\d+|\d+", alloy)
    compositions = [float(i) for i in compositions]
    # print('Compositions: ', compositions)

    elements_and_compositions = dict(zip(elements, compositions))
    s_mix = 0
    h_mix = 0

    for i in elements_and_compositions:
        s_mix = s_mix + (elements_and_compositions[i] / 100) * (math.log((elements_and_compositions[i] / 100)))
        h_mix = h_mix + (elements_and_compositions[i] / 100) * parameters[i]['enthalphy']
    s_mix = -1*s_mix

    big, small = diff(elements)
    # print("big atoms : ", big)
    # print("small atoms : ", small)
    delta_d = (comps(elements_and_compositions,big) - comps(elements_and_compositions,small)) / (comps(elements_and_compositions,big))
    delta_e = (electro(elements_and_compositions,big) - electro(elements_and_compositions,small)) / (electro(elements_and_compositions,big) + electro(elements_and_compositions,small))
    try:
        return h_mix, s_mix, delta_d, delta_e, elements[np.argmax(compositions)]
    except:
        print("Error: ", elements, compositions)
        return None, None, None, None, None

def make_params_df(df):
    h_mix_values = []
    s_mix_values = []
    delta_e_values = []
    delta_d_values = []
    for index, row in df.iterrows():
        h_mix, s_mix, delta_d, delta_e, max_element = prepare_params(row["bmg_alloy"])
        if h_mix != None and s_mix != None and delta_d != None and delta_e != None:
            pass
        else:
            print("SOMETHING IS WRONG WITH ALLOY")
        h_mix_values.append(h_mix)
        s_mix_values.append(s_mix)
        delta_d_values.append(delta_d)
        delta_e_values.append(delta_e)
    
    df['h_mix'] = h_mix_values
    df['s_mix'] = s_mix_values
    df['delta_d'] = delta_d_values
    df['delta_e'] = delta_e_values
    return df

def alloy_to_1d_tensor(alloy_str, alloy_max_len = alloy_max_len):
    tensor = torch.zeros(alloy_max_len)
    elements, compositions = get_elements_and_compositions(alloy_str)
    i = 0
    for idx in range(0, len(elements) + len(compositions), 2):
        tensor[idx] = element_to_index(elements[i])
        tensor[idx + 1] = compositions[i]
        i += 1
    return tensor

def elements_to_1d_tensor(elements, element_max_len = alloy_max_len // 2):
    tensor = torch.zeros(element_max_len)
    for idx in range(len(elements)):
        tensor[idx] = element_to_index(elements[idx])
    # print(tensor)
    return tensor

def composition_to_1d_tensor(compositions, composition_max_len = alloy_max_len // 2):
    tensor = torch.zeros(composition_max_len)
    for idx in range(len(compositions)):
        tensor[idx] = compositions[idx]
    # print(tensor)
    return tensor

def tensor_to_elements(tensor):
    elements = []
    for idx in range(len(tensor)):
        elements.append(index_to_element(int(tensor[idx])))
    return elements
