class GroundTruthSizeError(ValueError):
    def __init__(self, ground_truth_size) -> None:
        super().__init__(f"GROUND_TRUTH_SIZE SHOULD BE (x, x, 2) or (x, x, 3). Found {ground_truth_size}")
