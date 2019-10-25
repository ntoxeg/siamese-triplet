
# coding: utf-8

# In[1]:


#  get_ipython().magic('load_ext autoreload')
#  get_ipython().magic('autoreload 2')


# In[2]:


from PIL import Image as PImage
from IPython.display import Image as NbImage, display

from fastai.vision import *
from torchvision import transforms

from data import ImageDataset
import numpy as np


# In[32]:


tfm = partial(crop_pad, size=512, padding_mode="zeros")

def embed(learner:Learner, img:Image):
    return learner.predict(tfm(copy(img)))[1]

def _embed_imgset(learner:Learner, imgset):
    return [embed(learner, img) for img in imgset]

def _find_nneighbours(learner:Learner, img:Image, embedlist, n=5):
    imgembed = embed(learner, img)
    scores = [torch.dot(imgembed-emb, imgembed-emb) for emb in embedlist]
    results = sorted(zip(scores, range(len(embedlist))))
    return list(zip(*results[:n]))[1]

def find_nneighbours(learner:Learner, img:Image, imgset, n=5):
    embedlist = _embed_imgset(learner, imgset)
    imgembed = embed(learner, img).cpu()
    scores = [F.cosine_similarity(imgembed, emb, dim=0) for emb in embedlist]
    results = sorted(zip(scores, range(len(imgset))), reverse=True)
    return list(zip(*results[:n]))[1]

def show_nneighbours(learner:Learner, img:Image, imgset, n=5, embedlist=None):
    if embedlist is None:
        embedlist = _embed_imgset(learner, imgset)
    idxs = _find_nneighbours(learner, img, embedlist, n)
    to_img = transforms.ToPILImage()
    images = [to_img(img_.data) for img_ in [imgset[i] for i in idxs]]
    display(*images)


# In[4]:

def load_data(data_dir, classes):
#  data_dir='../data'
    #  data_split='val'
    #  classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    n_epochs = 1
    emsize = 32
    batch_size = 2
    margin = 3.


# In[5]:


# Parameters
    n_epochs = 100
    batch_size = 8
    emsize = 128


# In[29]:


    tfms = transforms.Compose([
        transforms.ToTensor(),
        Image,
    #     partial(crop_pad, size=512, padding_mode="zeros")
    ])
    dataset = ImageDataset(
        f"{data_dir}/",
        classes,
        tfms=tfms
    )
    n_classes = len(classes)

    imgset = [img for img, label in dataset]
    print(len(dataset))
    return imgset


# In[7]:

def load_default_learner():
    return load_learner("../siamese", "export.pkl").to_fp32()


# In[12]:


#  embedlist = _embed_imgset(learner, imgset)
# np.save("classy_coconut_embeddings_val.npz", torch.stack(embedlist).numpy())


# In[33]:


#  img = tfms(PImage.open("images/000000008021-head.jpg"))
#  show_nneighbours(learner, img, imgset, embedlist=embedlist)

