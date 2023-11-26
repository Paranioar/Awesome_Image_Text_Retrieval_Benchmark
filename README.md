# Image_Text_Retrieval_Benchmark
PyTorch implementation for Module Collection of Image-Text Retrieval for Further Exploration.

Importantly, the code (*completed in September 2022*) is not comprehensive and is not executable directly.  
It functions as a compilation of popular modules, designed to ease adaptation for other domains.

## Call for Contributors

*We welcome any improvements and supplements by **pulling requests** to enhance the functionality of this code. Feel free to **promote and share your papers** during this collaborative process.*


## Structure and Location

### [Holistic Feature Aggregation](https://github.com/Paranioar/Image_Text_Retrieval_Benchmark/tree/main/lib/modules/aggregation.py)
Basic Aggregation, Sequential GRU, Global Attention, Generalized Pooling, etc.  

### [Cross-Modality Interaction](https://github.com/Paranioar/Image_Text_Retrieval_Benchmark/tree/main/lib/modules/interaction.py)
Like-Cosine Attention, Focal Attention, Relation-wise Attention,  
Recurrent Attention, Transformer Attention, Bilinear Attention, etc.  

### [Similarity Construction](https://github.com/Paranioar/Image_Text_Retrieval_Benchmark/tree/main/lib/modules/similarity.py)
*Scalar Representation:* Inner-product Similarity, Order-embedding Similarity, etc.  
*Vector Representation:* Block-based Similarity, Symmetric-based or Asymmetric-based Similarity, etc.  
*Graph-based Aggregation:* Local Alignments Enhancement, Global Alignments Guidance, etc.     
*Attention-based Aggregation:* Local Alignments Filtration, Guidance Alignments Aggregation, etc.   

### [Objective Function](https://github.com/Paranioar/Image_Text_Retrieval_Benchmark/tree/main/lib/modules/lossfunction.py)
Birank Loss, CMPL Loss, Binary Cross-entropy Loss, Angular Loss, etc.

## Reference

If this code is useful for your research, please cite the relative papers in [Awesome_Cross_Modal_Pretraining_Transfering](https://github.com/Paranioar/Awesome_Cross_Modal_Pretraining_Transfering/blob/main/conventional_method.md).

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).  
