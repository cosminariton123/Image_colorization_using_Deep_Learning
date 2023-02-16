class GroundTruthSizeError(ValueError):
    def __init__(self, ground_truth_size) -> None:
        super().__init__(f"GROUND_TRUTH_SIZE should be (x, x, 2) or (x, x, 3). Found {ground_truth_size}")

class NoiseDeviationValueError(ValueError):
    def __init__(self, noise_deviation) -> None:
        super().__init__(f"NOISE_DEVIATION should be in interval [0, 999]. Found {noise_deviation}")

class NoisePercentageValueError(ValueError):
    def __init__(self, NOISE_PERCENTAGE) -> None:
        super().__init__(f"NOISE_PERCENTAGE should be in interval [0, 1]. Found {NOISE_PERCENTAGE}")
