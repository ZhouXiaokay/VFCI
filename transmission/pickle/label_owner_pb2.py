# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: label_owner.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11label_owner.proto\"A\n\rmiddle_output\x12\r\n\x05round\x18\x01 \x01(\x05\x12\r\n\x05\x65poch\x18\x02 \x01(\x05\x12\x12\n\nparams_msg\x18\x03 \x01(\x0c\".\n\x0bmiddle_grad\x12\r\n\x05round\x18\x01 \x01(\x05\x12\x10\n\x08grad_msg\x18\x02 \x01(\x0c\x32>\n\x11LabelOwnerService\x12)\n\ttop_fp_bp\x12\x0e.middle_output\x1a\x0c.middle_gradb\x06proto3')



_MIDDLE_OUTPUT = DESCRIPTOR.message_types_by_name['middle_output']
_MIDDLE_GRAD = DESCRIPTOR.message_types_by_name['middle_grad']
middle_output = _reflection.GeneratedProtocolMessageType('middle_output', (_message.Message,), {
  'DESCRIPTOR' : _MIDDLE_OUTPUT,
  '__module__' : 'label_owner_pb2'
  # @@protoc_insertion_point(class_scope:middle_output)
  })
_sym_db.RegisterMessage(middle_output)

middle_grad = _reflection.GeneratedProtocolMessageType('middle_grad', (_message.Message,), {
  'DESCRIPTOR' : _MIDDLE_GRAD,
  '__module__' : 'label_owner_pb2'
  # @@protoc_insertion_point(class_scope:middle_grad)
  })
_sym_db.RegisterMessage(middle_grad)

_LABELOWNERSERVICE = DESCRIPTOR.services_by_name['LabelOwnerService']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MIDDLE_OUTPUT._serialized_start=21
  _MIDDLE_OUTPUT._serialized_end=86
  _MIDDLE_GRAD._serialized_start=88
  _MIDDLE_GRAD._serialized_end=134
  _LABELOWNERSERVICE._serialized_start=136
  _LABELOWNERSERVICE._serialized_end=198
# @@protoc_insertion_point(module_scope)
