# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: foxglove/LocationFix.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1a\x66oxglove/LocationFix.proto\x12\x08\x66oxglove\x1a\x1fgoogle/protobuf/timestamp.proto\"\xca\x02\n\x0bLocationFix\x12-\n\ttimestamp\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x10\n\x08\x66rame_id\x18\x07 \x01(\t\x12\x10\n\x08latitude\x18\x01 \x01(\x01\x12\x11\n\tlongitude\x18\x02 \x01(\x01\x12\x10\n\x08\x61ltitude\x18\x03 \x01(\x01\x12\x1b\n\x13position_covariance\x18\x04 \x03(\x01\x12N\n\x18position_covariance_type\x18\x05 \x01(\x0e\x32,.foxglove.LocationFix.PositionCovarianceType\"V\n\x16PositionCovarianceType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x10\n\x0c\x41PPROXIMATED\x10\x01\x12\x12\n\x0e\x44IAGONAL_KNOWN\x10\x02\x12\t\n\x05KNOWN\x10\x03\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'foxglove.LocationFix_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _LOCATIONFIX._serialized_start=74
  _LOCATIONFIX._serialized_end=404
  _LOCATIONFIX_POSITIONCOVARIANCETYPE._serialized_start=318
  _LOCATIONFIX_POSITIONCOVARIANCETYPE._serialized_end=404
# @@protoc_insertion_point(module_scope)