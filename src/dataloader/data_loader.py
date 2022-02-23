import numpy as np
import json
import torch
from torch.utils.data import Dataset


class CallUtterance_Dataset(Dataset):
    
    def __init__(self, tokenizer, call_transcript_df, model_config):
        self.tokenizer = tokenizer
        self.call_transcript_df = call_transcript_df
        self.model_config = model_config
        
    def __len__(self):
        return self.call_transcript_df.shape[0]
    
    def __getitem__(self, idx):
        call_transcript = self.call_transcript_df.iloc[idx]['utterance'].split('\n')
        sent_idx = self.call_transcript_df.iloc[idx]['label_index']
        target = self.call_transcript_df.iloc[idx]['call_reason']
        marked = self.call_transcript_df.iloc[idx]['markable']
        
        if marked != 1:
            marked = 0
            
        sent = call_transcript[sent_idx]
        
        encoding = self.tokenizer.encode_plus(
                                sent,
                                max_length=self.model_config['max_seq_len'],
                                truncation=True,
                                add_special_tokens=True, 
                                return_token_type_ids=True,
                                pad_to_max_length=True,
                                return_attention_mask=True)

        ids = encoding['input_ids']
        masks = encoding['attention_mask']
        token_type_ids = encoding['token_type_ids']
        
        return {'ids': torch.tensor(ids, dtype=torch.long),
                'masks': torch.tensor(masks, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(target, dtype=torch.long),
                'marked': torch.tensor(marked, dtype=torch.long)}


class CallTranscript_Dataset(Dataset):
    
    def __init__(self, tokenizer, call_transcript_df, model_config,
                 num_class, label2id):
        self.tokenizer = tokenizer
        self.call_transcript_df = call_transcript_df
        self.model_config = model_config
        self.num_class = num_class
        self.label2id = label2id
        
    def __len__(self):
        return self.call_transcript_df.shape[0]
    
    def __getitem__(self, idx):
        call_transcript = self.call_transcript_df.iloc[idx]['call_transcript'].split('\n')
        call_index = self.call_transcript_df.iloc[idx]['call_index']
        
        heir_ids = np.zeros((self.model_config['max_sents'],
                             self.model_config['max_seq_len']))
        heir_masks = np.zeros((self.model_config['max_sents'],
                               self.model_config['max_seq_len']))
        heir_token_type_ids = np.zeros((self.model_config['max_sents'],
                                        self.model_config['max_seq_len']))
        targets = np.zeros((self.model_config['max_sents'], self.num_class))
        marked_sents = np.zeros(self.model_config['max_sents'])
        
        heir_utt_bar = np.zeros((self.model_config['max_sents'], self.num_class))
        utt_bar = np.array(json.loads(self.call_transcript_df.iloc[idx]['utt_bar']))
        
        for i, utt in enumerate(call_transcript):
            if i < self.model_config['max_sents']:
                encoding = self.tokenizer.encode_plus(
                                        utt,
                                        max_length=self.model_config['max_seq_len'],
                                        truncation=True,
                                        add_special_tokens=True, 
                                        return_token_type_ids=True,
                                        pad_to_max_length=True,
                                        return_attention_mask=True)

                heir_ids[i] = encoding['input_ids']
                heir_masks[i] = encoding['attention_mask']
                heir_token_type_ids[i] = encoding['token_type_ids']
                targets[i][self.label2id['No Label']] = 1
                
                heir_utt_bar[i] = utt_bar[i]
   
        target = self.call_transcript_df.iloc[idx]['call_label']
        
        if call_index < self.model_config['max_sents']:
            targets[call_index, target] = 1
            targets[call_index, self.label2id['No Label']] = 0
            
            marked_sents[call_index] = 1
            heir_utt_bar[call_index] = np.zeros(self.num_class)
            heir_utt_bar[call_index][target] = 1

        return {'ids': torch.tensor(heir_ids, dtype=torch.long),
                'masks': torch.tensor(heir_masks, dtype=torch.long),
                'token_type_ids': torch.tensor(heir_token_type_ids, dtype=torch.long),
                'targets': torch.tensor(targets, dtype=torch.long),
                'marked': torch.tensor(marked_sents, dtype=torch.long),
                'utt_bar': torch.tensor(heir_utt_bar, dtype=torch.float)
                }
    