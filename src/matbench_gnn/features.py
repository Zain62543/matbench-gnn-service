"""Composition-based Magpie features (baseline representation)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def magpie_features(structures: Sequence) -> pd.DataFrame:
    """Featurise pymatgen structures with matminer's Magpie preset.

    Only the chemical composition is used, so this baseline ignores geometry.
    That is deliberate: it is the reference a structure-aware GNN has to beat.
    """
    # Imported lazily so the rest of the package (metrics, plots) stays light.
    from matminer.featurizers.composition import ElementProperty

    featurizer = ElementProperty.from_preset("magpie")
    featurizer.impute_nan = True
    compositions = [s.composition for s in structures]
    rows = featurizer.featurize_many(compositions, ignore_errors=True, pbar=False)
    return pd.DataFrame(rows, columns=featurizer.feature_labels())
