import automatic_annotator_pb2_grpc
import grpc
from time import sleep
from concurrent import futures
from embedder import *
from semantic_search import *
from PIL import Image as PImage
import hydra
import logging


class AutoAnnotateServicer(automatic_annotator_pb2_grpc.AutoAnnotateServicer):
    def Annotate(self, request, context):
        imgset = load_data(request.datapath)
        exemplar = PImage.open(request.expath)
        learner = load_default_learner()
        bbox_search = BBoxSimilaritySearch(learner)
        ans = []
        for img in imgset:
            ans.extend(bbox_search(exemplar, img))

        ans_bytes = [img.tobytes() for img in ans]
        return automatic_annotator_pb2_grpc.AnnotateResponse(proposal=ans_bytes)


log = logging.getLogger(__name__)

@hydra.main()
def serve(cfg):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    automatic_annotator_pb2_grpc.add_AutoAnnotateServicer_to_server(
        AutoAnnotateServicer(),
        server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    log.info("The server is running...")


if __name__ == "__main__":
    serve()
    while True:
        sleep(1)
