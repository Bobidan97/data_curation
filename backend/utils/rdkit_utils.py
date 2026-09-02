"""rdkit_utils.py — Shared low-level RDKit helpers.

Small primitives that were previously copy-pasted across curation.py,
desirability.py, chemical_space.py, splits.py and the frontend image helpers.
Centralising them removes duplication and gives one place to tune behaviour.

Everything here degrades gracefully: a bad/None SMILES yields None rather than
raising, and the structural-alert catalogue returns None if RDKit is absent.
"""
from __future__ import annotations

# Module-level singletons (built lazily, reused across calls)
_ALERT_CATALOG = None


def mol_from_smiles(smiles):
    """Parse a SMILES string to an RDKit Mol, or return None on any failure."""
    if not isinstance(smiles, str):
        return None
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def murcko_scaffold(smiles: str) -> str | None:
    """Return the canonical Bemis–Murcko scaffold SMILES for a molecule.

    Strips side chains, keeping the ring systems plus their connecting linkers,
    then canonicalises. Returns None for invalid/unparseable input.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return None

    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return None


def alert_catalog():
    """Return a cached RDKit FilterCatalog with PAINS + Brenk + NIH + ZINC.

    These are the standard published structural-alert sets (promiscuous,
    reactive, and otherwise undesirable substructures). Built once and reused.
    Returns None if RDKit is unavailable.
    """
    global _ALERT_CATALOG
    if _ALERT_CATALOG is None:
        try:
            from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
        except ImportError:
            return None
        params = FilterCatalogParams()
        for cat in (
            FilterCatalogParams.FilterCatalogs.PAINS,
            FilterCatalogParams.FilterCatalogs.BRENK,
            FilterCatalogParams.FilterCatalogs.NIH,
            FilterCatalogParams.FilterCatalogs.ZINC,
        ):
            params.AddCatalog(cat)
        _ALERT_CATALOG = FilterCatalog(params)
    return _ALERT_CATALOG


def bonds_within_atom_set(mol, atom_set) -> list[int]:
    """Return bond indices whose both endpoints lie in ``atom_set``.

    Used when highlighting a substructure: an atom highlight looks cleaner when
    the bonds entirely inside the highlighted region are coloured too.
    """
    atoms = set(atom_set)
    return [
        b.GetIdx() for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms
    ]
