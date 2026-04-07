import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, name: str, role: str = "primary") -> None:
        self.name = name
        self.role = role

    def required_columns(self) -> list[str]:
        """Return columns required by this metric."""
        return []

    def validate(self, values: np.ndarray) -> np.ndarray:
        """Validate or transform extracted values."""
        return values

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> np.ndarray:
        """Extract metric values from dataframe."""
        raise NotImplementedError("This has not been implemented")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, role={self.role!r})"