"""
Personal Pharmacopoeia — SVG Export Script
Exports each molecule as a clean SVG file using RDKit.
No titles or labels — pure skeletal structures only.

Requirements:
    pip install rdkit

Usage:
    python export_molecules.py

Output:
    molecules/       — one SVG per drug, named by drug
    all_molecules.svg — all structures on one canvas, ready to scatter

Intentionally excluded (no conventional SMILES):
    - Filgrastim (G-CSF)    — 175-amino acid protein
    - Dalteparin            — polysaccharide
    - Heparin               — polysaccharide
    - L-Asparaginase        — enzyme
    - Pegaspargase          — PEGylated enzyme
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import os

DRUGS = [
    # Chemotherapy — first line
    ("vincristine",         "CCC1(CC2CC(C3=C(CN(C2)C1)C4=CC=CC=C4N3)(C(=O)OC)O)C(=O)OC"),
    ("dexamethasone",       "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@@]4(C)[C@H]3[C@@H](O)C[C@@]2(C)[C@@]1(O)C(=O)CO"),
    ("methotrexate",        "CN(CC1=CN=C2N=C(N)N=C(N)C2=N1)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O"),
    ("cyclophosphamide",    "ClCCNP1(=O)OCCCN1CCCl"),
    ("cytarabine",          "NC1=NC(=O)N(C=C1)[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O"),
    ("doxorubicin",         "COc1cccc2C(=O)c3c(O)c4C[C@@](O)(C(=O)CO)C[C@H](O[C@H]5C[C@H](N)[C@H](O)[C@H](C)O5)[C@@H]4c3C(=O)c12"),
    ("6-mercaptopurine",    "Sc1ncnc2[nH]cnc12"),
    # Chemotherapy — relapse
    ("mitoxantrone",        "NCCNC1=CC2=C(C=C1)C(=O)C1=C(O)C=CC=C1C2=O"),
    ("etoposide",           "COc1cc([C@@H]2c3cc4c(cc3[C@@H](O)[C@@H]2CO)OCO4)cc(OC)c1OC"),
    ("hydrocortisone",      "C[C@@H]1CC[C@H]2[C@@H](C1)[C@@H]3CC[C@@]4(O)[C@H](CCC4=O)[C@@]3(C)C[C@H]2O"),
    # Antiemetics
    ("ondansetron",         "CC1=CN=CN1CC2=C(C3=CC=CC=C3N4CCCCC4=O)C=CC=C2"),
    ("cyclizine",           "CN1CCN(CC1)C(c1ccccc1)c1ccccc1"),
    ("prochlorperazine",    "CN1CCN(CCCN2c3ccccc3Sc3ccc(Cl)cc32)CC1"),
    ("metoclopramide",      "CCN(CC)CCNC(=O)c1cc(Cl)c(N)c(OC)c1"),
    ("levomepromazine",     "CN(C)CCCC1c2ccccc2Sc2ccc(OC)cc12"),
    ("domperidone",         "O=C1NC2=CC=CC=C2N1CCCN3CCC(CC3)N4C(=O)c5ccccc5N4"),
    # GI / gastric protection
    ("lansoprazole",        "CC1=CN=C(CS(=O)C2=NC3=CC=CC=C3N2)C(OCC(F)(F)F)=C1"),
    ("ranitidine",          "CNC(=C[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1"),
    # Antimicrobials
    ("sulfamethoxazole",    "CC1=CC(NS(=O)(=O)C2=CC=C(N)C=C2)=NO1"),
    ("trimethoprim",        "COC1=C(OC)C(CC2=CN=C(N)N=C2N)=CC(OC)=C1"),
    ("metronidazole",       "CC1=NC=C(CCO)N1CC=O"),
    ("clarithromycin",      "CC[C@@H]1OC(=O)[C@H](C)[C@@H](O[C@@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](OC(=O)[C@@H](C)O)[C@@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O[C@H]3[C@@H](N(C)C)[C@H](O)[C@@H](O)[C@H](C)O3)O1"),
    ("amoxicillin",         "CC1(C)S[C@@H]2[C@H](NC(=O)[C@@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O"),
    ("clavulanic_acid",     "OC(=O)[C@@H]1N2C(=O)[C@@H](COC=C)[C@@H]2OC1=O"),
    # Analgesia
    ("morphine",            "OC1=CC2=C3[C@H]4CC[C@@](O)(CC4=C[C@@H]3OCC2)N(C)C1"),
    ("codeine",             "COc1ccc2c(c1)C[C@@H]1[C@@H]3CC=C[C@H](O)[C@@H]3N(C)CC1O2"),
    ("dihydrocodeine",      "COC1=CC2=C3[C@@H]4CC[C@@](O)(CC4)[C@@H]3OCC2=C1N(C)CC"),
    ("paracetamol",         "CC(=O)Nc1ccc(O)cc1"),
    ("ibuprofen",           "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("diclofenac",          "OC(=O)Cc1ccccc1Nc2c(Cl)cccc2Cl"),
    # Benzodiazepines
    ("diazepam",            "CN1C(=O)CN=C(C2=CC=CC=C2)C3=C1C=CC(=C3)Cl"),
    ("lorazepam",           "C1CN=C(C2=CC=CC=C2Cl)C3=C(N1C(=O)O)C=CC(=C3)Cl"),
    ("clonazepam",          "O=C1CN=C(c2ccccc2Cl)c2cc([N+](=O)[O-])ccc2N1"),
    # Local anaesthetics (EMLA)
    ("lidocaine",           "CCN(CC)CC(=O)Nc1c(C)cccc1C"),
    ("prilocaine",          "CCCNC(C)C(=O)Nc1ccccc1C"),
    # Mental health
    ("fluoxetine",          "CNCCC(c1ccccc1)Oc1ccc(cc1)C(F)(F)F"),
    ("sertraline",          "CN[C@@H]1CC[C@@H](c2ccc(Cl)c(Cl)c2)c2ccccc12"),
    ("citalopram",          "CN(C)CCCN1C(=O)c2ccc(F)cc2C1=Cc1ccc(cn1)C#N"),
    # Anticonvulsant
    ("phenytoin",           "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1"),
    # Antihistamine
    ("cetirizine",          "OC(=O)CN1CCN(CC1)C(c1ccccc1)c1ccc(Cl)cc1"),
    # Supportive
    ("mesna",               "OCS(=O)(=O)O"),
    ("leucovorin",          "NC1=NC2=C(N=C1)NCC(N2)CNC1=CC=C(C=C1)C(=O)NC(CCC(=O)O)C(=O)O"),
    ("allopurinol",         "O=C1NC2=NC=NC2=N1"),
]

SIZE      = 400
PADDING   = 0.08
LINE_W    = 2.0
MONOCHROME = False


def make_drawer(w, h):
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = d.drawOptions()
    opts.padding = PADDING
    opts.bondLineWidth = LINE_W
    opts.addStereoAnnotation = False
    opts.addAtomIndices = False
    opts.explicitMethyl = False
    if MONOCHROME:
        opts.useBWAtomPalette()
    return d


def mol_to_svg(smiles, name, size=SIZE):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  x  Could not parse: {name}")
        return None
    AllChem.Compute2DCoords(mol)
    d = make_drawer(size, size)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return d.GetDrawingText()


# ── Individual SVGs ──────────────────────────────────────────────
out_dir = "molecules"
os.makedirs(out_dir, exist_ok=True)

svgs = {}
for name, smiles in DRUGS:
    print(f"  Rendering {name}...")
    svg = mol_to_svg(smiles, name)
    if svg:
        with open(os.path.join(out_dir, f"{name}.svg"), "w") as f:
            f.write(svg)
        svgs[name] = svg

print(f"\n  {len(svgs)} SVGs saved to ./{out_dir}/")


# ── Combined canvas ──────────────────────────────────────────────
COLS   = 8
CELL   = SIZE + 40
ROWS_N = (len(svgs) + COLS - 1) // COLS
CW     = COLS * CELL
CH     = ROWS_N * CELL

lines = [
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{CW}" height="{CH}" viewBox="0 0 {CW} {CH}">',
    f'  <rect width="{CW}" height="{CH}" fill="white"/>',
]

for i, (name, smiles) in enumerate(DRUGS):
    if name not in svgs:
        continue
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    d = make_drawer(SIZE, SIZE)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    inner = d.GetDrawingText()
    start = inner.find(">", inner.find("<svg")) + 1
    end   = inner.rfind("</svg>")
    content = inner[start:end].strip()
    col = i % COLS
    row = i // COLS
    x   = col * CELL
    y   = row * CELL
    lines.append(f'  <g transform="translate({x},{y})" id="{name}">')
    lines.append(f'    {content}')
    lines.append('  </g>')

lines.append('</svg>')

with open("all_molecules.svg", "w") as f:
    f.write("\n".join(lines))

print(f"  Combined canvas saved to ./all_molecules.svg")
print("\nOpen all_molecules.svg in Illustrator or Inkscape.")
print("Each molecule is a <g id='name'> group — select, scatter, rotate freely.")