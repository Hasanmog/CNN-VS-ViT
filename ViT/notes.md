 # Why we add patch embedding with positional embedding and not multiply? 
 
   
**Addition**: When positional embeddings are added to patch embeddings, the original information in the patch embeddings is preserved. The addition operation allows the model to retain the full context of the image patch while incorporating positional information. This is crucial for the transformer to understand the spatial relationship between patches.

**Multiplication**: Element-wise multiplication could distort the original patch embeddings, as the operation is more invasive. Multiplication could scale down or nullify parts of the embedding, making it harder for the model to learn from the original patch information.