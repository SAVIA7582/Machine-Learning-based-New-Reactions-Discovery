from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdChemReactions


def to_reaction(smiles):
    """
    Convert reaction smiles to reaction object.
    :param smiles: smiles of reaction.
    :return: reaction object if the smiles is valid otherwise None
    """
    try:
        return rdChemReactions.ReactionFromSmarts(smiles)
    except:
        return None


def to_lim_int_seq(reaction, target_length):
    """
    Convert reaction object to integer sequence with limited length
    :param reaction: reaction object
    :param target_length: length to limit
    :return: the integer sequence
    """
    def to_int_seq():

        def sanitize_molecule(mol):
            try:
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol, clearAromaticFlags=True)
                return mol
            except Exception:
                return None

        result = []
        reactants = [sanitize_molecule(mol) for mol in reaction.GetReactants()]
        products = [sanitize_molecule(mol) for mol in reaction.GetProducts()]
        for mol in reactants + products:
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            atom_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            bond_list = [int(bond.GetBondTypeAsDouble()) for bond in mol.GetBonds()]
            result.extend(atom_list + bond_list)
            Chem.RemoveHs(mol)
        return result

    def limit_length(seq, length):
        if len(seq) < length:
            seq.extend([0.0] * (length - len(seq)))
        else:
            seq = seq[:length]
        return seq

    return limit_length(to_int_seq(), target_length)


def encode_properties(reaction):

    """
    calculate properties of reactant and product of a reaction, and combine them to a list
    :param reaction: reaction object
    :return: the list of properties of reactant and product of the reaction
    """

    def reactant():
        return reaction.GetReactants()[0]

    def product():
        return reaction.GetProducts()[0]

    def properties(mol):
        """
        calculate the properties of a molecular.
        :param mol: molecular object.
        :return: list of the properties of given molecular
        """
        properties = {
            'molecular_weight': Descriptors.MolWt(mol),
            'polarity': Descriptors.TPSA(mol),
            'solubility': Descriptors.MolLogP(mol),
            'pKa': Descriptors.MolMR(mol),
            'electronegativity': Descriptors.NumRadicalElectrons(mol)
        }
        return list(properties.values())

    return properties(reactant()) + properties(product())

