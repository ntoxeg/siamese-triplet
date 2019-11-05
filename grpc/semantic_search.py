#!/usr/bin/env python
# coding: utf-8

# In[91]:


#  get_ipython().run_line_magic('load_ext', 'autoreload')
#  get_ipython().run_line_magic('autoreload', '2')


# In[92]:


from embedder import *
import numpy as np


# In[104]:


class BBoxSimilaritySearch(nn.Module):
    def __init__(self, learner):
        super().__init__()
        self.learner = learner
        self.tfms = transforms.Compose([
            partial(crop_pad, size=512, padding_mode="zeros")
        ])
        self.toimg = transforms.ToPILImage()
        self.totsr = transforms.ToTensor()
        
    def forward(self, exemplar:Image, search_target:Image, n=5, iters=10):
#         exe_embed = self.embedder(exemplar)
#         target_embed = self.embedder(search_target)
#         ans = self.searcher(exe_embed, target_embed)
        # hmm, maybe one backbone and multiple heads?
        # would Centernet itself work well here?
        # concat CenterNet features with Siamese?
        # recurrent bbox controller?
        #  adaptive computation time?
        # higher order training?
        w, h = exemplar.size[1], exemplar.size[0]
        
        x_prior = torch.distributions.Dirichlet(torch.ones(search_target.size[1]))
        y_prior = torch.distributions.Dirichlet(torch.ones(search_target.size[0]))
        x_probs = x_prior.sample()
        y_probs = y_prior.sample()
        
        xs = map(int, torch.multinomial(x_probs, iters, replacement=True))
        ys = map(int, torch.multinomial(y_probs, iters, replacement=True))
        windows = [(x, y, x+w, y+h) for x, y in zip(xs, ys)]
        search_target_pil = self.toimg(search_target.data)
        crops = list(map(lambda bbox: Image(self.totsr(search_target_pil.crop(bbox))), windows))
        idxs = find_nneighbours(self.learner, exemplar, crops, n=n)
        
        return [windows[i] for i in idxs]


# In[ ]:


#  embedlist = _embed_imgset(learner, imgset)


# In[105]:


#  bbox_search = BBoxSimilaritySearch(learner)


# In[120]:

def get_proposals(bbox_search, imgpath, targetpath):
    img = tfms(PImage.open(imgpath))
    target = tfms(PImage.open(targetpath))
    with torch.no_grad():
        return bbox_search(img, target)

