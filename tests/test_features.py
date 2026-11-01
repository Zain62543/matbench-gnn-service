import pytest

pytest.importorskip("matminer")
pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from matbench_gnn.features import magpie_features  # noqa: E402


def _rocksalt(a: float, cation: str, anion: str) -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def test_magpie_shape_and_finite():
    structures = [_rocksalt(4.2, "Mg", "O"), _rocksalt(5.64, "Na", "Cl")]
    df = magpie_features(structures)

    assert len(df) == 2
    assert df.shape[1] > 100  # Magpie preset: many element-property statistics
    assert df.notna().all().all()


def test_features_depend_on_composition():
    df = magpie_features([_rocksalt(4.2, "Mg", "O"), _rocksalt(5.64, "Na", "Cl")])
    assert not df.iloc[0].equals(df.iloc[1])
