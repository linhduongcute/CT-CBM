from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, \
      BertModel, DebertaTokenizer, DebertaModel, GPT2Model, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM 
    #   ModernBertModel

from cbm_models import ModelXtoCtoY_function

import os
from pathlib import Path
import torch


def load_project_env(env_path=None):
    """Load simple KEY=VALUE entries from the project .env file."""
    if env_path is None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_hf_token():
    load_project_env()
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "Missing Hugging Face token. Add HF_TOKEN=your_token to a .env file "
            "at the project root, or set HUGGINGFACE_HUB_TOKEN in your environment."
        )
    return hf_token


def load_tokenizer(config):
    if config.model_name == 'roberta-base':
        return RobertaTokenizer.from_pretrained(config.model_name)
    elif config.model_name == 'roberta-large':
        return RobertaTokenizer.from_pretrained(config.model_name)
    elif config.model_name == 'bert-base-uncased':
        return BertTokenizer.from_pretrained(config.model_name)
    elif config.model_name == 'deberta-base':
        return DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    elif config.model_name == 'deberta-large':
        return DebertaTokenizer.from_pretrained('microsoft/deberta-large')
    elif config.model_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    elif config.model_name == 'gemma':
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=get_hf_token())
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    raise ValueError(f"Unsupported model_name for tokenizer: {config.model_name}")


# Load the model and tokenizer and bottleneck layer
def load_model_and_tokenizer(config, n_concepts = 4):
    """
    n_concepts : nombre de concepts dans le modèle joint (techniquement juste le nombre de neurones dans ModelXtoCtoY_function)
    """
    if config.model_name == 'roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        model = RobertaModel.from_pretrained(config.model_name)
    elif config.model_name == 'roberta-large':
        tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        model = RobertaModel.from_pretrained(config.model_name)
    elif config.model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(config.model_name)
        model = BertModel.from_pretrained(config.model_name)
    elif config.model_name == 'deberta-base':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        model = DebertaModel.from_pretrained('microsoft/deberta-base')
    elif config.model_name == 'deberta-large':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-large')
        model = DebertaModel.from_pretrained('microsoft/deberta-large')
    elif config.model_name == 'gpt2':
        model = GPT2Model.from_pretrained(config.model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
    elif config.model_name == 'gemma':
        hf_token = get_hf_token()
        tokenizer = load_tokenizer(config)
        model_kwargs = {
            "device_map": "auto" if torch.cuda.device_count() > 1 else {"": 0},
            "token": hf_token,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["max_memory"] = {
                i: f"{max(1, int(torch.cuda.mem_get_info(i)[0] / (1024 ** 3)) - 2)}GiB"
                for i in range(torch.cuda.device_count())
            }
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            **model_kwargs,
        )
        model_device_map = getattr(model, "hf_device_map", None)
        model = model.base_model
        if model_device_map is not None:
            model.hf_device_map = model_device_map
    elif config.model_name == 'lstm':
        # Implement the LSTM model setup here
        pass

    # load the bottleneck layer
    if config.model_name == 'lstm':
        ModelXtoCtoY_layer = ModelXtoCtoY_function(
            concept_classes=config.num_each_concept_classes,
            label_classes=config.num_labels,
            n_attributes=n_concepts,
            bottleneck=True,
            expand_dim=config.expand_dim,
            n_class_attr=config.num_each_concept_classes,
            use_relu=False,
            use_sigmoid=False,
            Lstm=True,
            aux_logits=config.is_aux_logits,
            config = config)
    else:
        ModelXtoCtoY_layer = ModelXtoCtoY_function(
            concept_classes= config.num_each_concept_classes,
            label_classes=config.num_labels,
            n_attributes= n_concepts,
            bottleneck=True,
            expand_dim=config.expand_dim,
            n_class_attr=config.num_each_concept_classes,
            use_relu=False,
            use_sigmoid=False,
            aux_logits=config.is_aux_logits,
            config = config)

    # load the linear classifier layer for the baseline (at origine it was for the differents strategy and then just define the loss of a sparse linear layer during training but we change this logique by creating a classs that encapsulate the linear classifier layer plus the loss in a same class in models.utils)

    if config.use_relu :
        # the following classifier is only used for basline modele so it sumulate a XtoCtoY architecture
        # if sigmoid after concept label (jointCBM then we need to compare the baseline it by putting a sigmoid here too)
        classifier = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, config.projection),
            nn.Dropout(config.dropout),
            torch.nn.Sigmoid(),
            torch.nn.Linear(config.projection, config.num_labels),        
       )
    elif config.use_relu:
        classifier = torch.nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, config.projection),     
            nn.Dropout(config.dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(config.projection, config.num_labels)       
        )
    else :
        classifier = nn.Sequential(
            torch.nn.Linear(model.config.hidden_size, config.num_labels)        
        )

    # model.config.hidden_size is a KEYWORD here : attribute of the model
    return model, tokenizer, ModelXtoCtoY_layer, classifier

# -------------------------- SPARSE LINEAR LAYER -------------

import torch
import torch.nn as nn

class RidgeLinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, l2_lambda):
        super(RidgeLinearLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def l2_penalty(self):
        return self.l2_lambda * torch.sum(self.linear.weight ** 2)

    def ridge_loss(self, outputs, targets):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets) + self.l2_penalty()
        return loss
    
class ElasticNetLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.01, l1_ratio=0.5):
        super(ElasticNetLinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
    def forward(self, x):
        return self.linear(x)
    
    def elasticnet_loss(self, outputs, targets):
        criterion = nn.CrossEntropyLoss()
        l1_norm = torch.norm(self.linear.weight, p=1)
        l2_norm = torch.norm(self.linear.weight, p=2)
        loss = criterion(outputs, targets) + self.alpha * (self.l1_ratio * l1_norm + (1 - self.l1_ratio) * l2_norm)
        return loss
    
    def reset_parameters(self):
        """Réinitialiser les paramètres de la couche linéaire."""
        self.linear.reset_parameters()
    
    def reset_residual_layer(self):
        """Réinitialiser la couche résiduelle (linear_layer)."""
        self.reset_parameters()
