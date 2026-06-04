"""Lightweight schema definitions for environmental observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel


class EnvironmentalObservation(BaseModel):
    """Row-level schema for an environmental in-situ observation."""

    station_id: str
    date: datetime
    latitude: float
    longitude: float


@dataclass(frozen=True)
class EnvironmentalObservationSchema:
    """Tabular schema definition for environmental observation datasets.

    Attributes:
        target_columns: One or more target variable columns expected in the dataset.
        date_column: Name of the datetime column.
        station_id_column: Name of the station identifier column.
        latitude_column: Name of the latitude column.
        longitude_column: Name of the longitude column.
    """

    target_columns: list[str]
    date_column: str = "date"
    station_id_column: str = "station_id"
    latitude_column: str = "latitude"
    longitude_column: str = "longitude"

    @property
    def required_columns(self) -> list[str]:
        """Return all required columns for a valid environmental dataset."""
        return [
            self.station_id_column,
            self.date_column,
            self.latitude_column,
            self.longitude_column,
            *self.target_columns,
        ]
