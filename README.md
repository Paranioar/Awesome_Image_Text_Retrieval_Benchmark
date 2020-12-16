# Image_Text_Matching_Bentchmark
PyTorch code for Bentchmark of Image Text Matching for Further Exploration. It builds on the open sources of latest SOTAs.  
We will release the codes that can be deployed on multi-GPU as soon as possible.  
## Code structure
### Generic representation extraction
For image, it has several options, e.g. commonly-used CNN, Transformer, various Graphs, Commonsense and Position insertion, etc.  
For caption, it includes commonly-used CNN and Bi-GRU, Transformer, several Graphs, and Commonsense insertion, etc.  
### Cross-attention interaction
Like-Cosine Attention, Focal Attention, Relation-wise Attention,  
Recurrent Attention, Transformer Attention, Bilinear Attention, etc.  
### Similarity construction
Scalar-based : Inner-product sim, Order-embedding sim, etc.  
vector-based : Block-based sim, Symmetric-based sim, Asymmetric-based sim, etc.  
### Similarity prediction  
Graph-based : Local alignments enhanced, Global alignments guided, etc.  
Attention-based: Local alignments filtration, Guidance alignments aggregation, etc.
### Loss function design  
Birank loss, CMPL loss, Binary cross-entropy loss,  Angular loss, etc.


