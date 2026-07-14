"""
Shared constants for Transition Metal Color Predictor
All physical/chemical parameters in one place
"""

# Spectrochemical series (relative ligand field strength)
# Source: Miessler, Fischer, Tarr - Inorganic Chemistry, 2014
LIGAND_STRENGTH = {
    'I':    0.60,
    'Br':   0.70,
    'Cl':   0.78,
    'SCN':  0.83,
    'NCS':  0.83,
    'F':    0.90,
    'H2O':  1.00,  # reference ligand
    'ox':   1.10,
    'acac': 1.15,
    'edta': 1.20,
    'dtc':  1.20,
    'NH3':  1.25,
    'dien': 1.30,
    'en':   1.40,
    'bipy': 1.50,
    'phen': 1.50,
    'dmg':  1.55,
    'NO2':  1.60,
    'CN':   1.70,
    'CO':   1.80,
}

# Coordination numbers by geometry
GEOMETRY_CN = {
    'octahedral':    6,
    'tetrahedral':   4,
    'square_planar': 4,
}

# Crystal field splitting factors relative to octahedral
GEOMETRY_FACTOR = {
    'octahedral':    1.00,
    'tetrahedral':   0.44,  # (4/9) * delta_oct
    'square_planar': 1.20,
}

# Atomic numbers of supported metals
METAL_Z = {
    'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
}

# d-electron count by (metal, oxidation_state)
D_ELECTRONS = {
    ('Ti', 3): 1, ('Ti', 2): 2,
    ('V',  3): 2, ('V',  2): 3,
    ('Cr', 3): 3, ('Cr', 2): 4,
    ('Mn', 2): 5, ('Mn', 3): 4,
    ('Fe', 2): 6, ('Fe', 3): 5,
    ('Co', 2): 7, ('Co', 3): 6,
    ('Ni', 2): 8, ('Ni', 3): 7,
    ('Cu', 2): 9, ('Cu', 1): 10,
}
def parse_ligand_strength(ligand_str: str) -> float:
    """
    Parse ligand string and return average field strength.
    
    Supports formats:
        'NH3'           -> single ligand
        'en+NH3'        -> mixed ligands (equal parts)
        'NH3*4+en*2'    -> mixed with counts
    
    Args:
        ligand_str: ligand specification string
    
    Returns:
        Average ligand field strength (float)
    """
    parts = ligand_str.split('+')
    total_strength = 0.0
    total_count = 0
    
    for part in parts:
        part = part.strip()
        
        if '*' in part:
            # Format: "NH3*4" or "en * 2"
            lig, cnt = part.split('*', 1)
            lig = lig.strip()
            cnt = int(cnt.strip())
        else:
            lig, cnt = part.strip(), 1
        
        strength = LIGAND_STRENGTH.get(lig, 1.0)
        total_strength += strength * cnt
        total_count += cnt
    
    return total_strength / max(total_count, 1)