from modeling import PreTrainedBertModel,BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import models
from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv,SAGEConv
import torch.nn.functional as F
from me_conv import MeConv
from MeGcn_conv import MeGCNConv
from simcse import simcse_sup_loss, simcse_unsup_loss, SimCSE_loss
from torch_geometric.utils import add_self_loops
#from model.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
from Orthographic_pytorch import Matrix, common_algorithm, Ortho_algorithm
import math
class BertForMeGCN(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super(BertForMeGCN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.vgg = VisionEncoder()
        #self.attention = MultiHeadAttention()
        self.attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8)
        self.batchnormal = nn.BatchNorm1d(config.hidden_size)
        self.linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.conv = GCNConv(config.hidden_size*2, config.hidden_size)
        self.meconv = MeConv(config.hidden_size, config.hidden_size)
        # self.meconv2 = MeConv(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,edge=None, image = None, labels=None):
        _, pooled_output,cls_pos,node_batch = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.dropout(pooled_output)
        device = _.device
        pooled_output = self.dropout(_)
        input_tensor = _[cls_pos]
        batch = self._get_batch(node_batch).to(device)#node_batch is node of news

        root_list = torch.nonzero(edge[1] == 0).squeeze().tolist()
        # if root_list.size(0) != input_ids.size(0):
        #     root_list = torch.cat([torch.tensor([0]).to(device), root_list + 1])
        text = input_tensor[root_list]
        image = self.vgg(image)
        image_tree = input_tensor
        image_tree[root_list] = image

        #image_tree graph conv
        image_tensors = self.meconv(image_tree, edge)  # shape (all_sents, hidden_size) input_tensor 10*768, edge 2*8
        # conv_tensors = F.relu(meconv_tensors)
        # conv_tensors = F.dropout(meconv_tensors, self.training)
        # conv_tensors = self.meconv2(input_tensor + conv_tensors,edge)
        image_tensors = F.relu(image_tensors)
        image_tensors = F.dropout(image_tensors, self.training)
        image_tensors = self.conv(torch.cat([image_tree, image_tensors], dim=-1), edge)
        # conv_tensors = F.relu(conv_tensors)
        #edge, _ = add_self_loops(edge)


        # text_tree graph conv

        meconv_tensors = self.meconv(input_tensor, edge) # shape (all_sents, hidden_size) input_tensor 10*768, edge 2*8
        # conv_tensors = F.relu(meconv_tensors)
        # conv_tensors = F.dropout(meconv_tensors, self.training)
        # conv_tensors = self.meconv2(input_tensor + conv_tensors,edge)
        conv_tensors = F.relu(meconv_tensors)
        conv_tensors = F.dropout(conv_tensors, self.training)
        conv_tensors = self.conv(torch.cat([input_tensor, conv_tensors], dim=-1),edge)
        # conv_tensors = F.relu(conv_tensors)


        pooled_output_image = scatter_mean(image_tensors, batch, dim=0)
        pooled_output = scatter_mean(conv_tensors, batch, dim=0)
        #pooled_output, weight = self.attention(image, pooled_output, pooled_output)
        even_number = [i for i in range(0, 2 * pooled_output.shape[0], 2)]
        odd_number = [i for i in range(1, 2 * pooled_output.shape[0] + 1, 2)]
        feature_stack = torch.empty(pooled_output.shape[0] * 2, pooled_output.shape[1]).to(device)
        feature_stack[odd_number] = pooled_output
        feature_stack[even_number] = pooled_output_image
        cse_loss = simcse_unsup_loss(feature_stack, device)

        multimodal_output = torch.cat([pooled_output_image,pooled_output],dim =1)
        multimodal_output = self.linear(multimodal_output)
        #pooled_output = common_algorithm(image, pooled_output)
        #pooled_output = self.batchnormal(pooled_output)
        #pooled_output = self.linear(pooled_output)
        logits = self.classifier(multimodal_output) + cse_loss

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss , image,text,multimodal_output
            #return loss, pooled_output_image, pooled_output, multimodal_output
        else:
            return logits,pooled_output_image,pooled_output,multimodal_output

    def _get_batch(self,batch):
        ans = []
        for i,count in enumerate(batch):
            ans+=[i]*count
        return torch.LongTensor(ans)


class BertForGCN(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super(BertForGCN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.conv = GCNConv(config.hidden_size, config.hidden_size)
        #self.conv = SAGEConv(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,edge=None, labels=None):
        _, pooled_output,cls_pos,node_batch = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.dropout(pooled_output)
        device = _.device
        pooled_output = self.dropout(_)
        input_tensor = _[cls_pos]
        batch = self._get_batch(node_batch).to(device)
        conv_tensors = self.conv(input_tensor, edge) # shape (all_sents, hidden_size)

        # conv_tensors = F.relu(conv_tensors)
        # conv_tensors = F.dropout(conv_tensors, self.training)
        pooled_output = scatter_mean(conv_tensors, batch, dim=0)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits,input_tensor

    def _get_batch(self,batch):
        ans = []
        for i,count in enumerate(batch):
            ans+=[i]*count
        return torch.LongTensor(ans)


class BertForMeGCNconv(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super(BertForMeGCNconv, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.conv = GCNConv(config.hidden_size, config.hidden_size)
        # self.meconv = MeConv(config.hidden_size, config.hidden_size)
        self.conv = MeGCNConv(config.hidden_size, config.hidden_size,add_self_loops=False)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,edge=None, labels=None):
        _, pooled_output,cls_pos,node_batch = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.dropout(pooled_output)
        device = _.device
        pooled_output = self.dropout(_)
        input_tensor = _[cls_pos]
        batch = self._get_batch(node_batch).to(device)
        conv_tensors = self.conv(input_tensor, edge) # shape (all_sents, hidden_size)
        # conv_tensors = self.conv(input_tensor,edge)
        # conv_tensors = F.relu(conv_tensors)
        # conv_tensors = F.dropout(conv_tensors, self.training)
        # conv_tensors = self.conv(conv_tensors,edge)
        
        pooled_output = scatter_mean(conv_tensors, batch, dim=0)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def _get_batch(self,batch):
        ans = []
        for i,count in enumerate(batch):
            ans+=[i]*count
        return torch.LongTensor(ans)


class BertOnly(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super(BertOnly, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.conv = GCNConv(config.hidden_size, config.hidden_size)
        # self.meconv = MeConv(config.hidden_size, config.hidden_size)
        
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,edge=None, labels=None):
        _, pooled_output,cls_pos,node_batch = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.dropout(pooled_output)
        device = _.device
        pooled_output = self.dropout(_)
        input_tensor = _[cls_pos]
        batch = self._get_batch(node_batch).to(device)
        # meconv_tensors = self.meconv(input_tensor, edge) # shape (all_sents, hidden_size)
        # conv_tensors = self.conv(input_tensor,edge)
        # conv_tensors = F.relu(conv_tensors)
        # conv_tensors = F.dropout(conv_tensors, self.training)
        # conv_tensors = self.conv(conv_tensors,edge)
        
        pooled_output = scatter_mean(input_tensor, batch, dim=0)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss,input_tensor
        else:
            return logits,input_tensor

    def _get_batch(self,batch):
        ans = []
        for i,count in enumerate(batch):
            ans+=[i]*count
        return torch.LongTensor(ans)


class BertForMe(PreTrainedBertModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels=2):
        super(BertForMeGCN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.conv = GCNConv(config.hidden_size*2, config.hidden_size)
        self.meconv = MeConv(config.hidden_size, config.hidden_size)
        # self.meconv2 = MeConv(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,edge=None, labels=None):
        _, pooled_output,cls_pos,node_batch = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        #pooled_output = self.dropout(pooled_output)
        device = _.device
        pooled_output = self.dropout(_)
        input_tensor = pooled_output[cls_pos]

        batch = self._get_batch(node_batch).to(device)
        
        meconv_tensors = self.meconv(input_tensor, edge) # shape (all_sents, hidden_size)
        # conv_tensors = F.relu(meconv_tensors)
        # conv_tensors = F.dropout(meconv_tensors, self.training)
        # conv_tensors = self.meconv2(input_tensor + conv_tensors,edge)
        # conv_tensors = F.relu(meconv_tensors)
        # conv_tensors = F.dropout(conv_tensors, self.training)
        # conv_tensors = self.conv(torch.cat([input_tensor, conv_tensors], dim=-1),edge)
        # conv_tensors = F.relu(conv_tensors)
        pooled_output = scatter_mean(meconv_tensors, batch, dim=0)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def _get_batch(self,batch):
        ans = []
        for i,count in enumerate(batch):
            ans+=[i]*count
        return torch.LongTensor(ans)

def self_attention(query, key, value, dropout=None, mask=None):
    """
    自注意力计算
    :param query: Q
    :param key: K
    :param value: V
    :param dropout: drop比率
    :param mask: 是否mask
    :return: 经自注意力机制计算后的值
    """
    d_k = query.size(-1)  # 防止softmax未来求梯度消失时的d_k
    # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    # 判断是否要mask，注：mask的操作在QK之后，softmax之前
    if mask is not None:
        """
        scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
        在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
        """
        # mask.cuda()
        # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引

        scores = scores.masked_fill(mask == 0, -1e9)

    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
    self_attn_softmax = scores
    # 判断是否要对相似概率分布进行dropout操作
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)

    # 注意：返回经自注意力计算后的值，以及进行softmax后的相似度（即相似概率分布）
    return torch.matmul(self_attn_softmax, value), self_attn_softmax


class MultiHeadAttention(nn.Module):
    """
    多头注意力计算
    """

    def __init__(self, head = 8, d_model = 768, dropout=0.1):
        """
        :param head: 头数
        :param d_model: 词向量的维度，必须是head的整数倍
        :param dropout: drop比率
        """
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)  # 确保词向量维度是头数的整数倍
        self.d_k = d_model // head  # 被拆分为多头后的某一头词向量的维度
        self.head = head
        self.d_model = d_model

        """
        由于多头注意力机制是针对多组Q、K、V，因此有了下面这四行代码，具体作用是，
        针对未来每一次输入的Q、K、V，都给予参数进行构建
        其中linear_out是针对多头汇总时给予的参数
        """
        self.linear_query = nn.Linear(d_model, d_model)  # 进行一个普通的全连接层变化，但不修改维度
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_softmax = None  # attn_softmax是能量分数, 即句子中某一个词与所有词的相关性分数， softmax(QK^T)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            """
            多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第二维（head维）添加一维，与后面的self_attention计算维度一样
            具体点将，就是：
            因为mask的作用是未来传入self_attention这个函数的时候，作为masked_fill需要mask哪些信息的依据
            针对多head的数据，Q、K、V的形状维度中，只有head是通过view计算出来的，是多余的，为了保证mask和
            view变换之后的Q、K、V的形状一直，mask就得在head这个维度添加一个维度出来，进而做到对正确信息的mask
            """
            mask = mask.unsqueeze(1)

        n_batch = query.size(0)  # batch_size大小，假设query的维度是：[10, 32, 512]，其中10是batch_size的大小

        """
        下列三行代码都在做类似的事情，对Q、K、V三个矩阵做处理
        其中view函数是对Linear层的输出做一个形状的重构，其中-1是自适应（自主计算）
        从这种重构中，可以看出，虽然增加了头数，但是数据的总维度是没有变化的，也就是说多头是对数据内部进行了一次拆分
        transopose(1,2)是对前形状的两个维度(索引从0开始)做一个交换，例如(2,3,4,5)会变成(2,4,3,5)
        因此通过transpose可以让view的第二维度参数变成n_head
        假设Linear成的输出维度是：[10, 32, 512]，其中10是batch_size的大小
        注：这里解释了为什么d_model // head == d_k，如若不是，则view函数做形状重构的时候会出现异常
        """
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]，head=8
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]

        # x是通过自注意力机制计算出来的值， self.attn_softmax是相似概率分布
        x, self.attn_softmax = self_attention(query, key, value, dropout=self.dropout, mask=mask)

        """
        下面的代码是汇总各个头的信息，拼接后形成一个新的x
        其中self.head * self.d_k，可以看出x的形状是按照head数拼接成了一个大矩阵，然后输入到linear_out层添加参数
        contiguous()是重新开辟一块内存后存储x，然后才可以使用.view方法，否则直接使用.view方法会报错
        """
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linear_out(x)

class VisionEncoder(nn.Module):

    def __init__(self, img_fc1_out=2742, img_fc2_out=768, dropout_p=0.4, fine_tune_module=True):
        super(VisionEncoder, self).__init__()
        self.fine_tune_module = fine_tune_module
        # 实例化
        vgg = models.vgg19(pretrained=True)
        vgg.classifier = nn.Sequential(*list(vgg.classifier.children())[:1])
        self.vis_encoder = vgg
        self.vis_enc_fc1 = torch.nn.Linear(4096, img_fc1_out)
        self.vis_enc_fc2 = torch.nn.Linear(img_fc1_out, img_fc2_out)
        self.dropout = nn.Dropout(dropout_p)
        self.fine_tune()

    def forward(self, images):
        """
        :参数: images, tensor (batch_size, 3, image_size, image_size)
        :返回: encoded images
        """
        x = self.vis_encoder(images)
        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )
        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc2(x))
        )
        return x

    def fine_tune(self):
        """
        允许或阻止vgg的卷积块2到4的梯度计算。
        """
        for p in self.vis_encoder.parameters():
            p.requires_grad = False

        # 如果进行微调，则只微调卷积块2到4
        for c in list(self.vis_encoder.children())[5:]:
            for p in c.parameters():
                p.requires_grad = self.fine_tune_module