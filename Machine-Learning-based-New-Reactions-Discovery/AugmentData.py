from rdkit import Chem
from rdkit.Chem import AllChem
import random
from rdkit.Chem import rdChemReactions

from pipe import Pipe
import pandas as pd


def random_delete_atom(molecular):
    """
    remove an atom from molecular randomly
    :param molecular: the molecular from which the atom is removed
    :return: the molecular after removing atom
    """
    def func():
        atoms = list(molecular.GetAtoms())
        if len(atoms) > 1:
            atom_to_remove = random.choice(atoms)
            mol = Chem.RWMol(molecular)
            mol.RemoveAtom(atom_to_remove.GetIdx())
            result = mol.GetMol()
        else:
            result = molecular

        return result

    return func


def random_insert_atom(mol, atom_symbol="C"):

    """
    insert an atom to a molecular.
    :param mol: molecular to insert
    :param atom_symbol: symbol of atom to be inserted
    :return: molecular after inserting
    """

    def func():
        rw_mol = Chem.RWMol(mol)
        atom_indices = list(range(rw_mol.GetNumAtoms()))
        bond_idx = random.choice(atom_indices)
        rw_mol.AddAtom(Chem.Atom(atom_symbol))
        rw_mol.AddBond(bond_idx, rw_mol.GetNumAtoms() - 1, Chem.rdchem.BondType.SINGLE)
        return rw_mol.GetMol()

    return func


def shuffle_smiles_to_mol(smiles):
    """
    randomly rearrange a smiles string to a new smiles then convert to molecular
    :param smiles: the smiles to be rearranged
    :return: molecular object from the rearranged smiles string
    """

    def shuffle_smiles(x):
        mol = Chem.MolFromSmiles(x)
        smiles_list = list(Chem.MolToSmiles(mol))
        random.shuffle(smiles_list)
        return ''.join(smiles_list)

    def func():
        return Chem.MolFromSmiles(shuffle_smiles(smiles))

    return func


def generate_isomers(mol):

    """
    Generate 10 isomers randomly
    :param mol: molecular object
    :return: list of isomers
    """

    isomers = set()
    for _ in range(10):
        new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        AllChem.EmbedMolecule(new_mol, randomSeed=random.randint(1, 1000))
        AllChem.UFFOptimizeMolecule(new_mol)
        isomers.add(Chem.MolToSmiles(new_mol))
    return list(isomers)


def smiles_augmentation(smiles, num=5):
    """
    generate augmented products by changing structure randomly
    :param smiles:
    :param num: times of augmentation
    :return: list of augmented products
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    func_list = [random_delete_atom(mol), random_insert_atom(mol), shuffle_smiles_to_mol(smiles)]

    augmented_smiles = {smiles}

    for _ in range(num):
        new_mol = random.choice(func_list)()
        if new_mol is not None:
            augmented_smiles.add(Chem.MolToSmiles(new_mol))

    augmented_smiles.update(generate_isomers(mol))

    return list(augmented_smiles)


def to_augment_data(original_smiles):
    """
    generate smiles of reactants and augmented products in given reaction represented by smiles
    :param original_smiles:
    :return:
    """
    try:
        reaction = rdChemReactions.ReactionFromSmarts(original_smiles, useSmiles=True)
        reactants = reaction.GetReactants()
        products = reaction.GetProducts()

        reactant_smiles = ".".join([Chem.MolToSmiles(reactant) for reactant in reactants])

        augmented_products = []
        for product in products:
            smiles = Chem.MolToSmiles(product)
            augmented_products.extend(smiles_augmentation(smiles))

        return original_smiles, reactant_smiles, augmented_products
    except Exception as e:
        print(f"Error processing reaction SMILES {original_smiles}: {e}")
        return "", "", []


def to_augment_dict(data):
    """
    generate dict of smiles
    :param data: 3 kinds of smiles(original reactions, reactant, augmented_products)
    :return: dictionary of smiles
    """

    def aug_dicts(original_smiles, reactant_smiles, augmented_products):

        def aug_dict(aug_p):
            return {
                "Original_SMILES": original_smiles,
                "Reactant_SMILES": reactant_smiles,
                "Augmented_SMILES": f"{reactant_smiles}>>{aug_p}"
            }

        return [aug_dict(aug_product) for aug_product in augmented_products]

    return aug_dicts(*data)


@Pipe
def to_df(data: list):
    return pd.DataFrame(data)