from typing import Dict, List, Union
import numpy as np


class KnobSpace:
    """Manages knob configuration space with separated tuning and physical spaces."""
    
    def __init__(
        self,
        knob_names: List[str],
        physical_ranges: Dict[str, tuple] = None,
        physical_defaults: Dict[str, Union[int, float, str]] = None,
    ):
        """Initialize with knob names and optional physical ranges.
        
        Args:
            knob_names: List of knob variable names
            physical_ranges: Dict mapping knob names to (min, max) tuples for execution
            physical_defaults: Dict mapping knob names to default physical values
        """
        self.knob_names = knob_names
        self.n_knobs = len(self.knob_names)
        self.name_to_idx = {name: idx for idx, name in enumerate(self.knob_names)}
        self._physical_ranges = {}
        self._default_vector = [0.5] * self.n_knobs
        if physical_ranges:
            self.add_physical_ranges(physical_ranges)
        if physical_defaults:
            self.add_physical_defaults(physical_defaults)

    @staticmethod
    def _normalize_physical_range(knob_name: str, physical_range: tuple) -> tuple[float, float]:
        min_val, max_val = physical_range
        min_val = float(min_val)
        max_val = float(max_val)
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            raise ValueError(f"Physical range for {knob_name} must be finite")
        if min_val <= 0.0 or max_val <= 0.0:
            raise ValueError(f"Physical range for {knob_name} must be positive for log-scale mapping")
        if min_val >= max_val:
            raise ValueError(f"Physical range for {knob_name} must satisfy min < max")
        return min_val, max_val
        
    @classmethod
    def from_search_space(cls, search_space: List[Dict[str, Union[str, float, int]]]):
        """Create KnobSpace from traditional search space definition."""
        knob_names = [knob['var'] for knob in search_space]
        physical_ranges = {knob['var']: (knob['min'], knob['max']) for knob in search_space}
        physical_defaults = {
            knob['var']: knob['default']
            for knob in search_space
            if 'default' in knob
        }
        return cls(knob_names, physical_ranges, physical_defaults)
        
    @property
    def dimension(self) -> int:
        return self.n_knobs
    
    def get_normalized_search_space(self) -> List[Dict[str, Union[str, float, int]]]:
        """Get normalized search space for LLM usage with 0-1 range."""
        return [
            {
                'var': knob_name,
                'min': 0.0,
                'max': 1.0,
                'default': self._default_vector[idx],
            }
            for idx, knob_name in enumerate(self.knob_names)
        ]
    
    def vector_to_config(self, vector: List[float]) -> Dict[str, float]:
        """Convert normalized vector [0,1]^n to physical knob configuration."""
        config = {}
        for i, knob_name in enumerate(self.knob_names):
            if i < len(vector):
                normalized_val = float(vector[i])
            else:
                normalized_val = self._default_vector[i]
            config[knob_name] = self._normalized_to_physical(knob_name, normalized_val)
        return config

    def _normalized_to_physical(self, knob_name: str, normalized_value: float) -> float:
        normalized_val = float(normalized_value)
        if not np.isfinite(normalized_val):
            raise ValueError(f"Normalized value for {knob_name} must be finite")
        if normalized_val < 0.0 or normalized_val > 1.0:
            raise ValueError(f"Normalized value for {knob_name} must be in [0, 1]")
        if knob_name in self._physical_ranges:
            min_val, max_val = self._physical_ranges[knob_name]
            physical_val = np.exp(
                normalized_val * (np.log(max_val) - np.log(min_val))
                + np.log(min_val)
            )
            return float(physical_val)
        return normalized_val
    
    def config_to_vector(self, config: Dict[str, Union[int, float, str]]) -> List[float]:
        """Convert physical knob configuration to normalized vector [0,1]^n."""
        vector = []
        for knob_name in self.knob_names:
            if knob_name in config:
                physical_val = float(config[knob_name])
                if knob_name in self._physical_ranges:
                    if physical_val <= 0.0 or not np.isfinite(physical_val):
                        raise ValueError(f"Physical value for {knob_name} must be positive and finite")
                    min_val, max_val = self._physical_ranges[knob_name]
                    # Convert from physical range to [0,1] using log scale
                    normalized_val = (np.log(physical_val) - np.log(min_val)) / (np.log(max_val) - np.log(min_val))
                    vector.append(max(0.0, min(1.0, normalized_val)))
                else:
                    # If no physical range, assume already normalized
                    if not np.isfinite(physical_val):
                        raise ValueError(f"Normalized value for {knob_name} must be finite")
                    vector.append(max(0.0, min(1.0, physical_val)))
            else:
                vector.append(self._default_vector[self.name_to_idx[knob_name]])
        return vector
    
    def get_default_vector(self) -> List[float]:
        """Get the normalized vector for the physical default configuration."""
        return list(self._default_vector)
    
    def validate_vector(self, vector: List[float]) -> bool:
        """Validate that vector has correct dimension and is in [0,1] range."""
        if len(vector) != self.n_knobs:
            return False
        try:
            values = [float(val) for val in vector]
        except (TypeError, ValueError):
            return False
        return all(np.isfinite(val) and 0.0 <= val <= 1.0 for val in values)
    
    def clamp_vector(self, vector: List[float]) -> List[float]:
        """Clamp vector values to [0,1], defaulting malformed coordinates."""
        clamped = []
        for value in vector:
            try:
                normalized_val = float(value)
            except (TypeError, ValueError):
                clamped.append(0.5)
                continue
            if not np.isfinite(normalized_val):
                clamped.append(0.5)
                continue
            clamped.append(max(0.0, min(1.0, normalized_val)))
        return clamped
    
    def get_knob_index(self, knob_name: str) -> int:
        """Get index of knob by name."""
        return self.name_to_idx[knob_name]
    
    def get_knob_name(self, index: int) -> str:
        """Get knob name by index."""
        return self.knob_names[index]
    
    def sample_random_vector(self) -> List[float]:
        """Sample random vector in [0,1]^n."""
        return np.random.random(self.n_knobs).tolist()
    
    def add_physical_ranges(self, physical_ranges: Dict[str, tuple]) -> None:
        """Add physical ranges for execution (called by SQLExecutor)."""
        for knob_name, physical_range in physical_ranges.items():
            self._physical_ranges[knob_name] = self._normalize_physical_range(
                knob_name,
                physical_range,
            )

    def add_physical_defaults(self, physical_defaults: Dict[str, Union[int, float, str]]) -> None:
        """Set default physical values and convert them to normalized x0."""
        for knob_name, physical_default in physical_defaults.items():
            if knob_name not in self.name_to_idx:
                continue
            self._default_vector[self.name_to_idx[knob_name]] = (
                self._normalize_default_value(knob_name, physical_default)
            )

    def _normalize_default_value(
        self,
        knob_name: str,
        physical_default: Union[int, float, str],
    ) -> float:
        default_value = float(physical_default)
        if not np.isfinite(default_value):
            raise ValueError(f"Default value for {knob_name} must be finite")

        if knob_name not in self._physical_ranges:
            if default_value < 0.0 or default_value > 1.0:
                raise ValueError(f"Normalized default for {knob_name} must be in [0, 1]")
            return default_value

        if default_value <= 0.0:
            raise ValueError(f"Default value for {knob_name} must be positive for log-scale mapping")
        min_val, max_val = self._physical_ranges[knob_name]
        normalized_val = (
            (np.log(default_value) - np.log(min_val))
            / (np.log(max_val) - np.log(min_val))
        )
        if normalized_val < 0.0 or normalized_val > 1.0:
            raise ValueError(f"Default value for {knob_name} must be inside its physical range")
        return float(normalized_val)
