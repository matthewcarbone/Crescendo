#!/usr/bin/env python3
'''Module for analyzing data within QM9'''

from rdkit import Chem


class structure:
    '''Functions for determining structures from a smiles input using rdkit chem.
    Current Structure types
        .ring -contains any ring
        .ring5 -contains a 5 atom ring
        .ring4 -contains a 4 atom ring
        .aromatic -contains an aromatic structure
        .doublebond -contains a double bond with the combinations of carbon, oxygen, and nitrogen
        .triplebond -contains a triple bond with the combinations of carbon and nitrogen
        .singlebond - does not contain .doublebond .triplebond and .aromatic
    
    Parameters
    ----------
    smiles : str
        smiles of target molecule as string
    
    Returns
    -------
    Boolean 
        the bool of a structure true if present false if not present
    Example
    -------
    #Molecule Benzene aromatic structure
    >>> structure('C1=CC=CC=C1').aromatic
    True

    '''
     
    def __init__(self,smiles):
        self.smile =smiles
        self.mol =Chem.MolFromSmiles(smiles)
        
    @property
    def ring(self):
        pattern = Chem.MolFromSmarts('[r]')
        return self.mol.HasSubstructMatch(pattern)
    
    @property
    def ring5(self):
        pattern = Chem.MolFromSmarts('[r5]')
        return self.mol.HasSubstructMatch(pattern)
    
    @property
    def ring4(self):
        pattern = Chem.MolFromSmarts('[r4]')
        return self.mol.HasSubstructMatch(pattern)
    
    @property
    def aromatic(self):
        pattern = Chem.MolFromSmarts('[a]')
        return self.mol.HasSubstructMatch(pattern)
    
    @property
    def doublebond(self):
        pattern1 = Chem.MolFromSmarts('C=C')
        pattern2 = Chem.MolFromSmarts('C=O')
        pattern3 = Chem.MolFromSmarts('C=N')
        pattern4 = Chem.MolFromSmarts('O=O')
        pattern5 = Chem.MolFromSmarts('O=N')
        pattern6 = Chem.MolFromSmarts('N=N')
        val= self.mol.HasSubstructMatch(pattern1) or self.mol.HasSubstructMatch(pattern2) or self.mol.HasSubstructMatch(pattern3) or self.mol.HasSubstructMatch(pattern4) or self.mol.HasSubstructMatch(pattern5) or self.mol.HasSubstructMatch(pattern6)  
        return val
    
    @property
    def triplebond(self):
        pattern1 = Chem.MolFromSmarts('C#C')
        pattern2 = Chem.MolFromSmarts('C#N')
        pattern3 = Chem.MolFromSmarts('N#N')
        val= self.mol.HasSubstructMatch(pattern1) or self.mol.HasSubstructMatch(pattern2) or self.mol.HasSubstructMatch(pattern3)
        return val

    @property
    def singlebond(self):
        return not self.doublebond and not self.triplebond and not self.aromatic
        
        


