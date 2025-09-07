# PHinKnot — Knot Geometry & PH Demos

Minimal scripts to explore geometric and persistent homology features for knots (trefoil demo).

## Files
- `pd_demo_modified.py` — Computes VR H1 Betti integral I(K) = sum of finite bar lengths; saves PD + Betti curve.
- `compare_trefoil.py` — Builds trefoil + “necklace” point cloud; computes RS(K), geometric IR(K), L/D; saves trefoil, PD, and raw/filtered β₁ curves.
- `github_python_functions.py` — Helper functions from the Celoria–Mahler paper.

## Requirements
Python 3.9+

## Credits / Upstream
This project incorporates code from:
- **knot-confinement-and-PH** (`github_python_functions.py`) by D. Celoria & B. I. Mahler,
  available at https://github.com/dceloriamaths/knot-confinement-and-PH
  and licensed under **GPL-2.0**.

## License
This project is distributed under the **GNU General Public License v2.0**.
See the `LICENSE` file for details.

## Reference
Celoria, D., & Mahler, B. I. (2022).
*A statistical approach to knot confinement via persistent homology.*
Proceedings of the Royal Society A.
Open-access: https://pmc.ncbi.nlm.nih.gov/articles/PMC9116441/
