from __future__ import annotations

from behavython.analysis.preprocess import preprocess_animal
from behavython.analysis.metrics import compute_roi_interaction
# from behavython.analysis.spatial import compute_spatial_metrics


def analyze_animal(animal, request) -> dict:
    """
    Main analysis entry point.

    Pipeline:
        1. Preprocess raw animal data → normalized numpy-based structure
        2. Compute interaction data (ROI, collisions) [optional]
        3. Compute movement metrics
        4. Compute exploration metrics
        5. Aggregate outputs

    Returns:
        dict: flat dictionary with all computed metrics
    """

    # -------------------------
    # 1. Preprocessing
    # -------------------------
    data = preprocess_animal(animal, request)

    # -------------------------
    # 2. ROI / interaction (optional for now)
    # -------------------------
    interaction_data = None
    interaction_data = compute_roi_interaction(data, request)

    # -------------------------
    # 3. Movement metrics (core, always present)
    # -------------------------
    movement_metrics = compute_movement_metrics(data, request)

    # -------------------------
    # 4. Exploration metrics (depends on interaction)
    # -------------------------
    exploration_metrics = {}

    if interaction_data is not None:
        exploration_metrics = compute_exploration_metrics(
            interaction_data,
            data,
            request,
        )

    # -------------------------
    # 5. Spatial metrics (optional / heavy)
    # -------------------------
    spatial_metrics = {}

    spatial_metrics = compute_spatial_metrics(data, request)

    # -------------------------
    # 6. Aggregate results
    # -------------------------
    results = {
        "animal_id": animal.id,

        # movement
        **movement_metrics,

        # exploration
        **exploration_metrics,

        # spatial
        **spatial_metrics,
    }

    return results