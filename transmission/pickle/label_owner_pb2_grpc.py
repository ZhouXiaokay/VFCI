# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import transmission.pickle.label_owner_pb2 as label__owner__pb2


class LabelOwnerServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.top_fp_bp = channel.unary_unary(
                '/LabelOwnerService/top_fp_bp',
                request_serializer=label__owner__pb2.middle_output.SerializeToString,
                response_deserializer=label__owner__pb2.middle_grad.FromString,
                )


class LabelOwnerServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def top_fp_bp(self, request, context):
        """AggregateServer provides the interface, client remotes the call
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LabelOwnerServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'top_fp_bp': grpc.unary_unary_rpc_method_handler(
                    servicer.top_fp_bp,
                    request_deserializer=label__owner__pb2.middle_output.FromString,
                    response_serializer=label__owner__pb2.middle_grad.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'LabelOwnerService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LabelOwnerService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def top_fp_bp(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/LabelOwnerService/top_fp_bp',
            label__owner__pb2.middle_output.SerializeToString,
            label__owner__pb2.middle_grad.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)