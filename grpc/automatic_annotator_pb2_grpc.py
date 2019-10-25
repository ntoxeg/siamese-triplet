# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import automatic_annotator_pb2 as automatic__annotator__pb2


class AutoAnnotateStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Annotate = channel.unary_unary(
        '/AutoAnnotate/Annotate',
        request_serializer=automatic__annotator__pb2.AnnotateRequest.SerializeToString,
        response_deserializer=automatic__annotator__pb2.AnnotateResponse.FromString,
        )


class AutoAnnotateServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Annotate(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_AutoAnnotateServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Annotate': grpc.unary_unary_rpc_method_handler(
          servicer.Annotate,
          request_deserializer=automatic__annotator__pb2.AnnotateRequest.FromString,
          response_serializer=automatic__annotator__pb2.AnnotateResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'AutoAnnotate', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
