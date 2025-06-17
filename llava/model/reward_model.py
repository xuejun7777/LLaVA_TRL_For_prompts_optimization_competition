from sentence_transformers import SentenceTransformer
import torch.nn as nn



class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.st_model = SentenceTransformer('/home/xuejun/LLaVA/sentence-transformer/all-MiniLM-L6-v2')
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        
        
    def forward(self, response_text, groundtruth_text):
        response_embedding = self.st_model.encode(response_text, convert_to_tensor=True)
        groundtruth_embedding = self.st_model.encode(groundtruth_text, convert_to_tensor=True)
        cosine_similarities = self.cosine_sim(response_embedding, groundtruth_embedding)
        reward = cosine_similarities
        for i, response in enumerate(response_text):
            if len(response) < 440:
                reward[i] += -1.5
            elif len(response) > 460:
                if cosine_similarities[i]>= 0.6:
                    reward[i] *= 2
                else:
                    reward[i] *= 1.6
            else:
                if cosine_similarities[i]>= 0.6:
                    reward[i] *= 1.5
                else:
                    reward[i] *= 1.2
                
        return reward