# Image_Text_Matching_Benchmark
PyTorch code for Benchmark of Image Text Matching that builds on the open sources for Further Exploration.  
We will release the codes that can be deployed on multi-GPU as soon as possible.  
## Code structure
### Generic representation extraction
For image, it has several options, e.g. commonly-used CNN, Transformer, various Graphs, Commonsense, etc.  
For caption, it includes commonly-used CNN and Bi-GRU, Transformer, several Graphs, Commonsense, etc.  
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
