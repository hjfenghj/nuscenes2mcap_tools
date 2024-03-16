"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
Generated by https://github.com/foxglove/schemas"""
import builtins
import foxglove.Color_pb2
import foxglove.Pose_pb2
import foxglove.Vector3_pb2
import google.protobuf.descriptor
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

@typing_extensions.final
class CylinderPrimitive(google.protobuf.message.Message):
    """A primitive representing a cylinder, elliptic cylinder, or truncated cone"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    POSE_FIELD_NUMBER: builtins.int
    SIZE_FIELD_NUMBER: builtins.int
    BOTTOM_SCALE_FIELD_NUMBER: builtins.int
    TOP_SCALE_FIELD_NUMBER: builtins.int
    COLOR_FIELD_NUMBER: builtins.int
    @property
    def pose(self) -> foxglove.Pose_pb2.Pose:
        """Position of the center of the cylinder and orientation of the cylinder. The flat face(s) are perpendicular to the z-axis."""
    @property
    def size(self) -> foxglove.Vector3_pb2.Vector3:
        """Size of the cylinder's bounding box"""
    bottom_scale: builtins.float
    """0-1, ratio of the diameter of the cylinder's bottom face (min z) to the bottom of the bounding box"""
    top_scale: builtins.float
    """0-1, ratio of the diameter of the cylinder's top face (max z) to the top of the bounding box"""
    @property
    def color(self) -> foxglove.Color_pb2.Color:
        """Color of the cylinder"""
    def __init__(
        self,
        *,
        pose: foxglove.Pose_pb2.Pose | None = ...,
        size: foxglove.Vector3_pb2.Vector3 | None = ...,
        bottom_scale: builtins.float = ...,
        top_scale: builtins.float = ...,
        color: foxglove.Color_pb2.Color | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["color", b"color", "pose", b"pose", "size", b"size"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["bottom_scale", b"bottom_scale", "color", b"color", "pose", b"pose", "size", b"size", "top_scale", b"top_scale"]) -> None: ...

global___CylinderPrimitive = CylinderPrimitive