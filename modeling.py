# coding=utf-8

# Reference: https://github.com/huggingface/pytorch-pretrained-BERT

"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function

import copy
import json
import math
from torch.nn import functional as F
import time
import six
import torch
from torchcrf import CRF
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import tensorflow as tf
import datetime
#from torch_multi_head_attention import MultiHeadAttention
from torch.autograd import Variable

def gelu(x):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
	"""Configuration class to store the configuration of a `BertModel`.
	"""
	def __init__(self,
				vocab_size,
				hidden_size=768,
				num_hidden_layers=12,
				num_attention_heads=12,
				intermediate_size=3072,
				hidden_act="gelu",
				hidden_dropout_prob=0.1,
				attention_probs_dropout_prob=0.1,
				max_position_embeddings=512,
				type_vocab_size=16,
				initializer_range=0.02):
		"""Constructs BertConfig.

		Args:
			vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
			hidden_size: Size of the encoder layers and the pooler layer.
			num_hidden_layers: Number of hidden layers in the Transformer encoder.
			num_attention_heads: Number of attention heads for each attention layer in
				the Transformer encoder.
			intermediate_size: The size of the "intermediate" (i.e., feed-forward)
				layer in the Transformer encoder.
			hidden_act: The non-linear activation function (function or string) in the
				encoder and pooler.
			hidden_dropout_prob: The dropout probabilitiy for all fully connected
				layers in the embeddings, encoder, and pooler.
			attention_probs_dropout_prob: The dropout ratio for the attention
				probabilities.
			max_position_embeddings: The maximum sequence length that this model might
				ever be used with. Typically set this to something large just in case
				(e.g., 512 or 1024 or 2048).
			type_vocab_size: The vocabulary size of the `token_type_ids` passed into
				`BertModel`.
			initializer_range: The sttdev of the truncated_normal_initializer for
				initializing all weight matrices.
		"""
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.hidden_act = hidden_act
		self.intermediate_size = intermediate_size
		self.hidden_dropout_prob = hidden_dropout_prob
		self.attention_probs_dropout_prob = attention_probs_dropout_prob
		self.max_position_embeddings = max_position_embeddings
		self.type_vocab_size = type_vocab_size
		self.initializer_range = initializer_range

	@classmethod
	def from_dict(cls, json_object):
		"""Constructs a `BertConfig` from a Python dictionary of parameters."""
		config = BertConfig(vocab_size=None)
		for (key, value) in six.iteritems(json_object):
			config.__dict__[key] = value
		return config

	@classmethod
	def from_json_file(cls, json_file):
		"""Constructs a `BertConfig` from a json file of parameters."""
		with open(json_file, "r") as reader:
			text = reader.read()
		return cls.from_dict(json.loads(text))

	def to_dict(self):
		"""Serializes this instance to a Python dictionary."""
		output = copy.deepcopy(self.__dict__)
		return output

	def to_json_string(self):
		"""Serializes this instance to a JSON string."""
		return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
	def __init__(self, config, variance_epsilon=1e-12):
		"""Construct a layernorm module in the TF style (epsilon inside the square root).
		"""
		super(BERTLayerNorm, self).__init__()
		self.gamma = nn.Parameter(torch.ones(config.hidden_size))
		self.beta = nn.Parameter(torch.zeros(config.hidden_size))
		self.variance_epsilon = variance_epsilon

	def forward(self, x):
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.variance_epsilon)
		return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
	def __init__(self, config):
		super(BERTEmbeddings, self).__init__()
		"""Construct the embedding module from word, position and token_type embeddings.
		"""
		self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
		self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
		self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

		# self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
		# any TensorFlow checkpoint file
		self.LayerNorm = BERTLayerNorm(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, input_ids, token_type_ids=None):
		seq_length = input_ids.size(1)
		position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
		position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		words_embeddings = self.word_embeddings(input_ids)
		position_embeddings = self.position_embeddings(position_ids)
		token_type_embeddings = self.token_type_embeddings(token_type_ids)

		embeddings = words_embeddings + position_embeddings + token_type_embeddings
		embeddings = self.LayerNorm(embeddings)
		embeddings = self.dropout(embeddings)
		return embeddings


class BERTSelfAttention(nn.Module):
	def __init__(self, config):
		super(BERTSelfAttention, self).__init__()
		if config.hidden_size % config.num_attention_heads != 0:
			raise ValueError(
				"The hidden size (%d) is not a multiple of the number of attention "
				"heads (%d)" % (config.hidden_size, config.num_attention_heads))
		self.num_attention_heads = config.num_attention_heads
		self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
		self.all_head_size = self.num_attention_heads * self.attention_head_size

		self.query = nn.Linear(config.hidden_size, self.all_head_size)
		self.key = nn.Linear(config.hidden_size, self.all_head_size)
		self.value = nn.Linear(config.hidden_size, self.all_head_size)

		self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attention_head_size)
		# Apply the attention mask is (precomputed for all layers in BertModel forward() function)
		attention_scores = attention_scores + attention_mask

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		return context_layer


class BERTSelfOutput(nn.Module):
	def __init__(self, config):
		super(BERTSelfOutput, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.LayerNorm = BERTLayerNorm(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BERTAttention(nn.Module):
	def __init__(self, config):
		super(BERTAttention, self).__init__()
		self.self = BERTSelfAttention(config)
		self.output = BERTSelfOutput(config)

	def forward(self, input_tensor, attention_mask):
		self_output = self.self(input_tensor, attention_mask)
		attention_output = self.output(self_output, input_tensor)
		return attention_output


class BERTIntermediate(nn.Module):
	def __init__(self, config):
		super(BERTIntermediate, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
		self.intermediate_act_fn = gelu

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class BERTOutput(nn.Module):
	def __init__(self, config):
		super(BERTOutput, self).__init__()
		self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
		self.LayerNorm = BERTLayerNorm(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class BERTLayer(nn.Module):
	def __init__(self, config):
		super(BERTLayer, self).__init__()
		self.attention = BERTAttention(config)
		self.intermediate = BERTIntermediate(config)
		self.output = BERTOutput(config)

	def forward(self, hidden_states, attention_mask):
		attention_output = self.attention(hidden_states, attention_mask)
		intermediate_output = self.intermediate(attention_output)
		layer_output = self.output(intermediate_output, attention_output)
		return layer_output


class BERTEncoder(nn.Module):
	def __init__(self, config):
		super(BERTEncoder, self).__init__()
		layer = BERTLayer(config)
		self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

	def forward(self, hidden_states, attention_mask):
		all_encoder_layers = []
		for layer_module in self.layer:
			hidden_states = layer_module(hidden_states, attention_mask)
			all_encoder_layers.append(hidden_states)
		return all_encoder_layers


class BERTPooler(nn.Module):
	def __init__(self, config):
		super(BERTPooler, self).__init__()
		self.dense = nn.Linear(config.hidden_size, config.hidden_size)
		self.activation = nn.Tanh()

	def forward(self, hidden_states):
		# We "pool" the model by simply taking the hidden state corresponding
		# to the first token.
		first_token_tensor = hidden_states[:, 0]
		#return first_token_tensor
		pooled_output = self.dense(first_token_tensor)
		pooled_output = self.activation(pooled_output)
		return pooled_output


class BertModel(nn.Module):
	"""BERT model ("Bidirectional Embedding Representations from a Transformer").

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

	config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
		num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

	model = modeling.BertModel(config=config)
	all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config: BertConfig):
		"""Constructor for BertModel.

		Args:
			config: `BertConfig` instance.
		"""
		super(BertModel, self).__init__()
		self.embeddings = BERTEmbeddings(config)
		self.encoder = BERTEncoder(config)
		self.pooler = BERTPooler(config)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None):
		if attention_mask is None:
			attention_mask = torch.ones_like(input_ids)
		if token_type_ids is None:
			token_type_ids = torch.zeros_like(input_ids)

		# We create a 3D attention mask from a 2D tensor mask.
		# Sizes are [batch_size, 1, 1, from_seq_length]
		# So we can broadcast to [batch_size, num_heads, to_seq_length, from_seq_length]
		# this attention mask is more simple than the triangular masking of causal attention
		# used in OpenAI GPT, we just need to prepare the broadcast dimension here.
		extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

		# Since attention_mask is 1.0 for positions we want to attend and 0.0 for
		# masked positions, this operation will create a tensor which is 0.0 for
		# positions we want to attend and -10000.0 for masked positions.
		# Since we are adding it to the raw scores before the softmax, this is
		# effectively the same as removing these entirely.
		extended_attention_mask = extended_attention_mask.float()
		extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

		embedding_output = self.embeddings(input_ids, token_type_ids)
		all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
		sequence_output = all_encoder_layers[-1]
		pooled_output = self.pooler(sequence_output)
		return all_encoder_layers, pooled_output

class BertForSequenceClassification(nn.Module):
	"""BERT model for classification.
	This module is composed of the BERT model with a linear layer on top of
	the pooled output.

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

	config = BertConfig(vocab_size=32000, hidden_size=512,
		num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

	num_labels = 2

	model = BertForSequenceClassification(config, num_labels)
	logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config, num_labels):
		super(BertForSequenceClassification, self).__init__()
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels)

		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		if labels is not None:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits, labels)
			return loss, logits
		else:
			return logits


class BertForQuestionAnswering(nn.Module):
	"""BERT model for Question Answering (span extraction).
	This module is composed of the BERT model with a linear layer on top of
	the sequence output that computes start_logits and end_logits

	Example usage:
	```python
	# Already been converted into WordPiece token ids
	input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
	input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
	token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

	config = BertConfig(vocab_size=32000, hidden_size=512,
		num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

	model = BertForQuestionAnswering(config)
	start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
	```
	"""
	def __init__(self, config):
		super(BertForQuestionAnswering, self).__init__()
		self.bert = BertModel(config)
		# TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.qa_outputs = nn.Linear(config.hidden_size, 2)

		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
		all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
		sequence_output = all_encoder_layers[-1]
		logits = self.qa_outputs(sequence_output)
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)

		if start_positions is not None and end_positions is not None:
			# If we are on multi-GPU, split add a dimension - if not this is a no-op
			start_positions = start_positions.squeeze(-1)
			end_positions = end_positions.squeeze(-1)
			# sometimes the start/end positions are outside our model inputs, we ignore these terms
			ignored_index = start_logits.size(1)
			start_positions.clamp_(0, ignored_index)
			end_positions.clamp_(0, ignored_index)

			loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
			start_loss = loss_fct(start_logits, start_positions)
			end_loss = loss_fct(end_logits, end_positions)
			total_loss = (start_loss + end_loss) / 2
			return total_loss
		else:
			return start_logits, end_logits


# BERT + softmax
class BertForTABSAJoint(nn.Module):
	def __init__(self, config, num_labels, num_ner_labels, max_seq_length):
		super(BertForTABSAJoint, self).__init__()
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		self.max_seq_length = max_seq_length
		# 初始化参数
		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	def forward(self, input_ids, token_type_ids, attention_mask, labels, ner_labels, ner_mask):
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)# 编码器输出的表示和池化层输出的表示
		# get the last hidden layer 获取序列中每个标记的表示
		sequence_output = all_encoder_layers[-1]
		# cross a dropout layer 随机失活
		sequence_output = self.dropout(sequence_output)
		pooled_output = self.dropout(pooled_output)
		# the Classifier of category & polarity
		logits = self.classifier(pooled_output)# 池化层输出是T[CLS],通过Linear得到P[CLS]
		ner_logits = self.ner_hidden2tag(sequence_output)# 编码器输出Tn,通过Linear得到Pn
		ner_logits.reshape([-1, self.max_seq_length, self.num_ner_labels])

		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits, labels)
		ner_loss_fct = CrossEntropyLoss(ignore_index=0)# 忽略0类
		ner_loss = ner_loss_fct(ner_logits.view(-1, self.num_ner_labels), ner_labels.view(-1))
		return loss, ner_loss, logits, ner_logits


# BERT + CRF
class BertForTABSAJoint_CRF(nn.Module):
	def __init__(self, config, num_labels, num_ner_labels):
		super(BertForTABSAJoint_CRF, self).__init__()
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		# CRF
		self.CRF_model = CRF(num_ner_labels, batch_first=True)

		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	def categorical_crossentropy_with_prior(y_pred, y_true, tau=2.0):
		prior = torch.tensor([1.0,38.0])
		log_prior = torch.log(prior+1e-8)
		for _ in range(y_pred.dim() - 1):
			log_prior = torch.unsqueeze(log_prior, 0)
		y_pred = y_pred - tau * log_prior.type(torch.FloatTensor).cuda()
		loss_fct = CrossEntropyLoss()
		return loss_fct(y_pred, y_true)


	def forward(self, input_ids, token_type_ids, attention_mask, labels, ner_labels, ner_mask, eval_flag):
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		# get the last hidden layer
		sequence_output = all_encoder_layers[-1]
		# cross a dropout layer
		sequence_output = self.dropout(sequence_output)
		pooled_output = self.dropout(pooled_output)
		# the Classifier of category & polarity
		logits = self.classifier(pooled_output)
		ner_logits = self.ner_hidden2tag(sequence_output)

		# the CRF layer of NER labels
		ner_loss_list = self.CRF_model(ner_logits, ner_labels, ner_mask.type(torch.ByteTensor).cuda(), reduction='none')
		ner_loss = torch.mean(-ner_loss_list)
		ner_predict = self.CRF_model.decode(ner_logits, ner_mask.type(torch.ByteTensor).cuda())

		# the classifier of category & polarity
		'''
		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits, labels)
		'''
		if eval_flag:#控制bool在测试的时候使用交叉熵 训练的时候使用互信息
			loss = BertForTABSAJoint_BiLSTM_CRF.categorical_crossentropy_with_prior(logits, labels, tau=2.0)
		else:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits, labels)
		
		return loss, ner_loss, logits, ner_predict

## 修改分界线---------------------------------------------------------------------------------------------------------------------
# BERT + BiLSTM + CRF
# 参考https://github.com/PeijiYang/BERT-BiLSTM-CRF-NER-pytorch/blob/master/models.py
class BertForTABSAJoint_BiLSTM_CRF(nn.Module):
	def __init__(self, config, num_labels, num_ner_labels):
		super(BertForTABSAJoint_BiLSTM_CRF, self).__init__()
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		#添加NER部分单独的dropout，防止过拟合--------------------------------------------修改成功
		self.dropoutNER = nn.Dropout(0.1)
		#------------------------------------------------------------------------------修改成功
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		# CRF
		self.CRF_model = CRF(num_ner_labels, batch_first=True)
		#初始dim=128，layer=1    最好结果为5层lstm，hidden_size=128，200效果不佳，原因可能是超过了潜在特征数量
		# 在每层lstm间加入dropout=0.5
		self.birnn = nn.LSTM(num_ner_labels, 128, num_layers=5, bidirectional=True, batch_first=True)
		# 尝试加入多头注意力机制--------------------------------------------------------修改
		# 参考https://github.com/yoseflaw/nerindo/blob/master/nerindo/models.py
		self.mutli_attention = nn.MultiheadAttention(embed_dim=num_ner_labels, num_heads=5, dropout=0.1)
		#-----------------------------------------------------------------------------修改
		# LSTM FC
		self.hidden2tag = nn.Linear(256, num_ner_labels)
		# 注意力 FC
		self.fc = nn.Linear(10, num_ner_labels)

		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	#增加互信息loss----------------------------------------------------------------	
	def categorical_crossentropy_with_prior(y_pred, y_true, tau=1.0):
		prior = torch.tensor([1.0,38.0])
		log_prior = torch.log(prior+1e-8)
		for _ in range(y_pred.dim() - 1):
			log_prior = torch.unsqueeze(log_prior, 0)
		y_pred = y_pred - tau * log_prior.type(torch.FloatTensor).cuda()
		loss_fct = CrossEntropyLoss()
		return loss_fct(y_pred, y_true)
	#----------------------------------------------------------------

	def forward(self, input_ids, token_type_ids, attention_mask, labels, ner_labels, ner_mask, eval_flag):# eval_flag
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		# get the last hidden layer
		sequence_output = all_encoder_layers[-1]
		# cross a dropout layer
		sequence_output = self.dropout(sequence_output)
		pooled_output = self.dropout(pooled_output)
		# the Classifier of category & polarity
		logits = self.classifier(pooled_output)
		ner_logits = self.ner_hidden2tag(sequence_output)#(24,128,5)
		# ner_logits传入BiLSTM中
		birnn_sequence_output, _ = self.birnn(ner_logits)
		birnn_sequence_output = self.dropoutNER(birnn_sequence_output)#(24,128,256)(batch, seq_len, num_directions * hidden_size)
		#----------------------------------------------------下面为修改
		'''
		# BiLSTM输出
		birnn_output = self.hidden2tag(birnn_sequence_output)#(24,128,5)
		# 多头注意力机制
		ner_logits = ner_logits.permute(1,0,2)
		attn_output, _ = self.mutli_attention(ner_logits, ner_logits,ner_logits)#(128,24,5)
		attn_output = attn_output.permute(1,0,2) #(24,128,5)
		# attn_output = self.hidden2tag(attn_output)#(24,128,5)
		# 拼接两者输出
		output = torch.cat((birnn_output,attn_output),2)
		output = self.fc(output)
		output = torch.tanh(output)
		# print(output.size())
		# print(ner_labels.size())#(24,128)
		# the CRF layer of NER labels
		ner_loss_list = self.CRF_model(output, ner_labels, ner_mask.type(torch.ByteTensor).cuda(), reduction='none')
		ner_loss = torch.mean(-ner_loss_list)
		ner_predict = self.CRF_model.decode(output, ner_mask.type(torch.ByteTensor).cuda())
		#---------------------------------------------------
		'''
		'''
		# 多头注意力机制------------------------------------------------------------------------------------修改
		ner_logits = ner_logits.permute(1,0,2)#(128,24,5)
		attn_output, _ = self.mutli_attention(ner_logits, ner_logits, ner_logits)#新增----------------------#(128,24,5)
		attn_output = attn_output.permute(1,0,2)#(24,128,5)
		# BiLSTM输出
		birnn_sequence_output, _ = self.birnn(attn_output)
		birnn_sequence_output = self.dropoutNER(birnn_sequence_output)#(24,128,256)
		birnn_output = self.hidden2tag(birnn_sequence_output)
		# the CRF layer of NER labels
		ner_loss_list = self.CRF_model(birnn_output, ner_labels, ner_mask.type(torch.ByteTensor).cuda(), reduction='none')
		ner_loss = torch.mean(-ner_loss_list)
		ner_predict = self.CRF_model.decode(birnn_output, ner_mask.type(torch.ByteTensor).cuda())
		#--------------------------------------------------------------------------------------------------修改
		'''
		
		# BiLSTM输出
		birnn_output = self.hidden2tag(birnn_sequence_output)
		# the CRF layer of NER labels
		ner_loss_list = self.CRF_model(birnn_output, ner_labels, ner_mask.type(torch.ByteTensor).cuda(), reduction='none')
		ner_loss = torch.mean(-ner_loss_list)
		ner_predict = self.CRF_model.decode(birnn_output, ner_mask.type(torch.ByteTensor).cuda())
		
		# the classifier of category & polarity 尝试采用focal loss
		# loss_fct = focal_loss(alpha=0.75, gamma=2, num_classes=2) 失败
		# loss_fct = DiceLoss(weight=[1,1])
		# 参考https://blog.csdn.net/HUSTHY/article/details/103887957对损失函数进行修改 权重分配yes：no = 38:1 失败
		# loss_fct = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([38,1])).float().cuda())
		
		if eval_flag:#控制bool在测试的时候使用交叉熵 训练的时候使用互信息
			loss = BertForTABSAJoint_BiLSTM_CRF.categorical_crossentropy_with_prior(logits, labels, tau=1.0)
		else:
			loss_fct = CrossEntropyLoss()
			loss = loss_fct(logits, labels)
		'''
		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits, labels)
		'''
		# loss = BertForTABSAJoint_BiLSTM_CRF.categorical_crossentropy_with_prior(logits, labels, tau=1.0)
		#----------------------------------------------------------------------修改
		return loss, ner_loss, logits, ner_predict

	
## ------------------------------------------------------------------------------------------------------------------------




#the model for ablation study, separate training
# BERT + softmax
class BertForTABSAJoint_AS(nn.Module):
	def __init__(self, config, num_labels, num_ner_labels, max_seq_length):
		super(BertForTABSAJoint_AS, self).__init__()
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		self.max_seq_length = max_seq_length

		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	def forward(self, input_ids, token_type_ids, attention_mask, labels, ner_labels, ner_mask):
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		# get the last hidden layer
		sequence_output = all_encoder_layers[-1]
		# cross a dropout layer
		sequence_output = self.dropout(sequence_output)
		pooled_output = self.dropout(pooled_output)
		# the Classifier of category & polarity
		logits = self.classifier(pooled_output)
		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits, labels)
		return loss, logits

#the model for ablation study, separate training
# BERT + softmax
class BertForTABSAJoint_T(nn.Module):
	def __init__(self, config, num_labels, num_ner_labels, max_seq_length):
		super(BertForTABSAJoint_T, self).__init__()
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		self.max_seq_length = max_seq_length

		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	def forward(self, input_ids, token_type_ids, attention_mask, labels, ner_labels, ner_mask):
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		# get the last hidden layer
		sequence_output = all_encoder_layers[-1]
		# cross a dropout layer
		sequence_output = self.dropout(sequence_output)
		pooled_output = self.dropout(pooled_output)
		# the Classifier of category & polarity
		ner_logits = self.ner_hidden2tag(sequence_output)
		ner_logits.reshape([-1, self.max_seq_length, self.num_ner_labels])

		ner_loss_fct = CrossEntropyLoss(ignore_index=0)
		ner_loss = ner_loss_fct(ner_logits.view(-1, self.num_ner_labels), ner_labels.view(-1))
		return ner_loss, ner_logits

#the model for ablation study, separate training
# BERT + CRF
class BertForTABSAJoint_CRF_AS(nn.Module):

	def __init__(self, config, num_labels, num_ner_labels):
		super(BertForTABSAJoint_CRF_AS, self).__init__()
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		# CRF
		self.CRF_model = CRF(num_ner_labels, batch_first=True)

		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	def forward(self, input_ids, token_type_ids, attention_mask, labels, ner_labels, ner_mask):
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		# get the last hidden layer
		sequence_output = all_encoder_layers[-1]
		# cross a dropout layer
		sequence_output = self.dropout(sequence_output)
		pooled_output = self.dropout(pooled_output)
		# the Classifier of category & polarity
		logits = self.classifier(pooled_output)
		# the classifier of category & polarity
		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits, labels)
		return loss, logits

#the model for ablation study, separate training
# BERT + CRF
class BertForTABSAJoint_CRF_T(nn.Module):
	def __init__(self, config, num_labels, num_ner_labels):
		super(BertForTABSAJoint_CRF_T, self).__init__()
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		# CRF
		self.CRF_model = CRF(num_ner_labels, batch_first=True)

		def init_weights(module):
			if isinstance(module, (nn.Linear, nn.Embedding)):
				# Slightly different from the TF version which uses truncated_normal for initialization
				# cf https://github.com/pytorch/pytorch/pull/5617
				module.weight.data.normal_(mean=0.0, std=config.initializer_range)
			elif isinstance(module, BERTLayerNorm):
				module.beta.data.normal_(mean=0.0, std=config.initializer_range)
				module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
			if isinstance(module, nn.Linear):
				module.bias.data.zero_()
		self.apply(init_weights)

	def forward(self, input_ids, token_type_ids, attention_mask, labels, ner_labels, ner_mask):
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		# get the last hidden layer
		sequence_output = all_encoder_layers[-1]
		# cross a dropout layer
		sequence_output = self.dropout(sequence_output)
		pooled_output = self.dropout(pooled_output)
		ner_logits = self.ner_hidden2tag(sequence_output)

		# the CRF layer of NER labels
		ner_loss_list = self.CRF_model(ner_logits, ner_labels, ner_mask.type(torch.ByteTensor).cuda(), reduction='none')
		ner_loss = torch.mean(-ner_loss_list)
		ner_predict = self.CRF_model.decode(ner_logits, ner_mask.type(torch.ByteTensor).cuda())

		return ner_loss, ner_predict

# 失败
'''
class focal_loss(nn.Module):
    """
    需要保证每个batch的长度一样，不然会报错。
    """
    def __init__(self,alpha=0.25,gamma = 2, num_classes = 2, size_average =True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi) = -α(1-yi)**γ * log(yi)
        :param alpha:
        :param gamma:
        :param num_classes:
        :param size_average:
        """
 
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            assert len(alpha) == num_classes
            print("Focal_loss alpha = {},对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.tensor(alpha)
        else:
            assert alpha<1 #如果α为一个常数,则降低第一类的影响
            #print("--- Focal_loss alpha = {},将对背景类或者大类负样本进行权重衰减".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)
        self.gamma = gamma
 
    def forward(self, preds,labels):
        """
        focal_loss损失计算
        :param preds: 预测类别. size:[B,N,C] or [B,C]  B:batch N:检测框数目 C:类别数
        :param labels: 实际类别. size:[B,N] or [B]
        :return:
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_softmax = F.softmax(preds,dim=1)
        preds_logsoft = torch.log(preds_softmax)
        # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
 
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax),self.gamma),preds_logsoft)
        loss = torch.mul(self.alpha,loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
'''
# dice loss https://github.com/shuaizzZ/Dice-Loss-PyTorch/blob/master/dice_loss.py
class DiceLoss(nn.Module):
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, 1, -1) # (N, 1, *)

        predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss
