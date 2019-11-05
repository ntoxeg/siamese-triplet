import automatic_annotator_pb2_grpc
import automatic_annotator_pb2
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
        imgset = load_data(request.datapath, ("person",))
        exemplar = tfms(PImage.open(request.expath))
        learner = load_default_learner()
        bbox_search = BBoxSimilaritySearch(learner)
#         ans = []
        msg = automatic_annotator_pb2.AnnotateResponse()
        for imgpath, img in imgset:
            msg.imgpaths.append(imgpath)
            
            proposals = automatic_annotator_pb2.Proposals()
            proposals.proposals.extend([
                automatic_annotator_pb2.Proposal(x1=x1,y1=y1,x2=x2,y2=y2) for x1,y1,x2,y2 in bbox_search(exemplar, img)
            ])
            msg.all_proposals.append(proposals)

#         ans_bytes = [img.tobytes() for img in ans]
#         msg.proposals.extend(ans_bytes)
        return msg


log = logging.getLogger(__name__)

@hydra.main()
def serve_and_sleep(cfg):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    automatic_annotator_pb2_grpc.add_AutoAnnotateServicer_to_server(
        AutoAnnotateServicer(),
        server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    log.info("The server is running...")
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve_and_sleep()
