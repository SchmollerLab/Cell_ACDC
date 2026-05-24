"""Shared type aliases for the domain model."""

from typing import NewType

FrameIndex = NewType('FrameIndex', int)
CellID = NewType('CellID', int)
ChannelName = NewType('ChannelName', str)
PixelSize = NewType('PixelSize', float)
