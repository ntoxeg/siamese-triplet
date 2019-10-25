import logging
import grpc

import automatic_annotator_pb2
import automatic_annotator_pb2_grpc

import hydra
from PIL import Image as PImage


channel = grpc.insecure_channel('localhost:50051')
stub = automatic_annotator_pb2_grpc.AutoAnnotateServiceStub(channel)

@hydra.main()
def annotator_client(cfg):
    print(cfg.pretty())
    request = automatic_annotator_pb2.AnnotateRequest(datapath=cfg.datapath, expath=cfg.expath)
    ret = stub.Annotate(request)
    ans = [PImage.frombytes(b) for b in ret.proposals]


if __name__ == "__main__":
    annotator_client()
