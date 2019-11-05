import grpc

import automatic_annotator_pb2
import automatic_annotator_pb2_grpc

import hydra
from PIL import Image as PImage
import json


channel = grpc.insecure_channel('localhost:50051')
stub = automatic_annotator_pb2_grpc.AutoAnnotateStub(channel)


# @hydra.main()
from flask import Flask
app = Flask(__name__)

@app.route('/annotate/<path:datapath>/<path:expath>')
def annotator_client(datapath, expath):
#     print(cfg.pretty())
    request = automatic_annotator_pb2.AnnotateRequest(
            datapath=datapath,
            expath=expath
    )
    ret = stub.Annotate(request)
    def open_image(i):
        return PImage.open(ret.imgpaths[i])
    ans = {
        imgpath: [
            (proposal.x1, proposal.y1, proposal.x2, proposal.y2) for proposal in ret.all_proposals[i].proposals
        ] for i, imgpath in enumerate(ret.imgpaths)
    }
    
    coco_dict = {
        "categories": [{"id": 100, "name": "head"}],
        "annotations": []
    }
    for i, (imgpath, proposals) in enumerate(ans.items()):
        for j, proposal in enumerate(proposals):
            img = open_image(i).crop(proposal)
            img.save(f"/home/adrian/projects/siamese-triplet/grpc/output/proposal_{i}_{j}.png")
            x1, y1, x2, y2 = proposal
            coco_dict["annotations"].append({"id": f"{i}_{j}", "class_id": 100, "bbox": [x1, y1, x2-x1, y2-y1]})
    with open("/home/adrian/projects/siamese-triplet/grpc/output/coco.json", "w") as f:
        json.dump(coco_dict, f)
    
    return "Proposals generated and saved to `output/`."

if __name__ == "__main__":
#     annotator_client()
    app.run()
