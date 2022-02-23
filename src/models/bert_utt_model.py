import torch.nn as nn
from transformers import BertModel
from transformers import BertPreTrainedModel


class BertModel_Utt(BertPreTrainedModel):
    
    def __init__(self, bert_config):
        super().__init__(bert_config)

        self.bert = BertModel(bert_config)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_config.hidden_size, 256)
        self.classifier_m = nn.Linear(bert_config.hidden_size, 64)
        self.marked_classifier = nn.Linear(64, 1)
        self.reason_classifier = nn.Linear(256, bert_config.num_labels)

        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        doc_positions=None,
    ):

        bertoutput = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        outputs = {}
        outputs['utt_rep'] = bertoutput['pooler_output']
        x = self.classifier(bertoutput['pooler_output'])
        outputs['logits'] = self.reason_classifier(x)
        y = self.classifier_m(bertoutput['pooler_output'])
        outputs['marked'] = self.marked_classifier(y)
            
        return outputs

