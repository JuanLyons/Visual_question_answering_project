# Visual_question_answering_project

The proposed method (Figure 3) is a simple model that combines the strong feature map encoders from pretrained CLIP. In this method, we have in parallel two inputs: the question and the image. Each one of them is fed into the corresponding CLIP encoder. After that, we obtain embeddings of the two inputs. Subsequently, we apply different strategies to join these two embedding vectors—concatenation or summation. Depending on the strategy chosen, we pass the output of that operation to an MLP, which will generate a probability distribution regarding whether the answer is yes or no. Therefore, our model fine-tunes pretrained CLIP embeddings in the closed-ended medical VQA task.

![Proposed method](images/Proposed_method.png)