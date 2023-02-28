import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification

from architecture.hierarchical_transformer import h_transformer

class HierachicalModel(nn.Module):
    
    def __init__(self, sentence_encoder_path = None, d_model: int = 768, nhead: int = 4, d_hid: int = 768,
                 nlayers: int = 1, dropout: float = 0.25):
        super().__init__()
        self.sentence_encoder = AutoModelForSequenceClassification.from_pretrained(sentence_encoder_path, local_files_only=True)
        self.h_transformer = h_transformer(d_model, nhead, d_hid, nlayers, dropout)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, batch_samples):
        # compute sentence encoding
        doc_encoding = None
        for i in range(batch_samples["input_ids"].shape[1]):
            sentence_encoder_output = self.sentence_encoder(batch_samples["input_ids"][:,i,:], batch_samples["attention_mask"][:,i,:], output_hidden_states=True)
            minibatch_encoding = sentence_encoder_output.hidden_states[-1][:,0,:]
            if doc_encoding is None:
                doc_encoding = minibatch_encoding.view(minibatch_encoding.shape[0], 1, minibatch_encoding.shape[1])
            else:
                doc_encoding = torch.cat((doc_encoding, minibatch_encoding.view(minibatch_encoding.shape[0], 1, minibatch_encoding.shape[1])), dim=1)
            print(doc_encoding.shape)

        # compute hierarchical attention mask to be used in the transformer at second level
        hierachical_attention_mask = batch_samples["attention_mask"]
        hierachical_attention_mask = hierachical_attention_mask.sum(dim=2)
        hierachical_attention_mask[hierachical_attention_mask > 1] = 1.0
        hierachical_attention_mask = hierachical_attention_mask.type(torch.FloatTensor)

        # compute hierarchical encoding
        hierarchical_encoding = self.h_transformer(doc_encoding, hierachical_attention_mask)
        print(hierarchical_encoding.shape)

        # average pooling
        hierarchical_encoding = hierarchical_encoding.mean(dim=1)

        # compute output
        output = self.fc(hierarchical_encoding)
        output = self.sigmoid(output)
        print(output)
        return output

