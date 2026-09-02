"""drawing.py — Render RDKit molecules to base64 PNG data URIs.

A single rendering primitive (``mol_to_uri``) plus SMILES-level convenience
wrappers, replacing five near-identical copies that had accumulated in the
frontend. All functions return a ``data:image/png;base64,...`` string ready for
``st.image`` / ``st.column_config.ImageColumn``, or ``None`` on any failure
(invalid SMILES, RDKit missing, draw error).

Uses RDKit's Cairo backend (MolDraw2DCairo) throughout for crisp output.
"""
from __future__ import annotations

import base64

# Standard highlight colours (RGB 0-1 floats)
AMBER = (0.99, 0.82, 0.30)   # SMARTS / substructure highlight
BLUE  = (0.22, 0.48, 0.93)   # SHAP atom highlight
CORAL = (1.00, 0.55, 0.45)   # activity-cliff R-group highlight


def mol_to_uri(
    mol,
    *,
    highlight_atoms=None,
    atom_colors=None,
    highlight_bonds=None,
    bond_colors=None,
    size: tuple = (400, 300),
    padding: float = 0.12,
    bond_line_width: int | None = None,
) -> str | None:
    """Render an RDKit Mol to a base64 PNG data URI with optional highlights.

    All highlight arguments are optional; pass an atom/bond index list plus a
    matching ``{idx: (r,g,b)}`` colour dict to highlight a region.
    """
    if mol is None:
        return None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(int(size[0]), int(size[1]))
        opts = drawer.drawOptions()
        opts.padding = padding
        if bond_line_width is not None:
            opts.bondLineWidth = bond_line_width
        rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_atoms or []),
            highlightAtomColors=atom_colors or {},
            highlightBonds=list(highlight_bonds or []),
            highlightBondColors=bond_colors or {},
        )
        drawer.FinishDrawing()
        return f"data:image/png;base64,{base64.b64encode(drawer.GetDrawingText()).decode()}"
    except Exception:
        return None


def render_smiles(
    smiles: str,
    *,
    size: tuple = (400, 300),
    padding: float = 0.12,
    bond_line_width: int | None = None,
) -> str | None:
    """Plain render of a SMILES string (no highlights)."""
    from utils.rdkit_utils import mol_from_smiles
    return mol_to_uri(
        mol_from_smiles(smiles),
        size=size, padding=padding, bond_line_width=bond_line_width,
    )


def render_smiles_atoms(
    smiles: str,
    atom_indices,
    color: tuple = BLUE,
    *,
    size: tuple = (240, 180),
    padding: float = 0.12,
) -> str | None:
    """Render a SMILES with a given set of atoms (and their internal bonds)
    highlighted in a single colour."""
    from utils.rdkit_utils import mol_from_smiles, bonds_within_atom_set
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    atoms = list(atom_indices)
    bonds = bonds_within_atom_set(mol, atoms)
    return mol_to_uri(
        mol,
        highlight_atoms=atoms,
        atom_colors={a: color for a in atoms},
        highlight_bonds=bonds,
        bond_colors={b: color for b in bonds},
        size=size,
        padding=padding,
    )


def render_smiles_smarts(
    smiles: str,
    smarts: str,
    color: tuple = AMBER,
    *,
    size: tuple = (400, 300),
) -> str | None:
    """Render a SMILES with every atom matching ``smarts`` highlighted.

    Falls back to a plain render when the SMARTS is empty/invalid.
    """
    from rdkit import Chem
    from utils.rdkit_utils import mol_from_smiles, bonds_within_atom_set

    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    pat = Chem.MolFromSmarts(smarts.strip()) if isinstance(smarts, str) and smarts.strip() else None
    if pat is None:
        return mol_to_uri(mol, size=size, padding=0.08, bond_line_width=2)

    atom_set: set[int] = set()
    for match in mol.GetSubstructMatches(pat):
        atom_set.update(match)
    atoms = sorted(atom_set)
    bonds = bonds_within_atom_set(mol, atoms)
    return mol_to_uri(
        mol,
        highlight_atoms=atoms,
        atom_colors={a: color for a in atoms},
        highlight_bonds=bonds,
        bond_colors={b: color for b in bonds},
        size=size,
        padding=0.08,
        bond_line_width=2,
    )
