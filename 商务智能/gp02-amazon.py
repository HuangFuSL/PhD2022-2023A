import functools
import itertools
import json
import lzma
import os
import typing as T

import gensim.corpora
import gensim.models
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import tqdm
import re
import transformers.models.bert.tokenization_bert

os.chdir(os.path.dirname(__file__))

C_BertModel = transformers.BertModel  # type: ignore
C_BertTokenizer = transformers.models.bert.tokenization_bert.BertTokenizer
C_DataLoader = torch.utils.data.DataLoader
C_Dictionary = gensim.corpora.Dictionary
C_LabelEncoder = sklearn.preprocessing.LabelEncoder
C_LDAModel = gensim.models.LdaModel
C_Module = torch.nn.Module
F_Padding = torch.nn.utils.rnn.pad_sequence
F_Split = sklearn.model_selection.train_test_split
T_Record = T.Dict[str, T.Any]
T_Sentence = T.List[T.List[str]]
T_Tensor = torch.Tensor
T_X = T.TypeVar('T_X')
T_Y = T.TypeVar('T_Y')

DATA_PATH = 'data-amazon.jsonl.xz'
PUNC_RE = re.compile(r'[!@#\$%\^&*()_+-=,./<>?;:\'"\[\]{}\\|`~]')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LV1_FIELDS = ['comments', 'hotCommentTagStatistics', 'productCommentSummary']
LV2_FIELDS = [
    'product/productId', 'product/title',
    'review/userId', 'review/summary', 'review/text', 'review/score'
]
LOADER_PARAMS = {
    'batch_size': 8,
    'shuffle': True
}
TRAIN_PARAMS = {
    'epochs': 1,
    'lr': 0.0003,
    'optim': torch.optim.Adam,
    'loss': torch.nn.CrossEntropyLoss,
    'weight': None
}

# ---------------------- MODEL DEFINITION --------------------------


class FM(C_Module):

    def __init__(self, embed_dim: int, feature_dims: T.List[int]):
        super().__init__()
        self.order_1_embed = torch.nn.ModuleList(
            torch.nn.Embedding(_, 1) for _ in feature_dims
        )
        self.order_2_embed = torch.nn.ModuleList(
            torch.nn.Embedding(_, embed_dim) for _ in feature_dims
        )

    def forward(self, Xf: T_Tensor):
        ''' Xf: Long tensor, shape of (N, field_num) '''
        order_1 = torch.cat([
            embed(Xf[:, i]) for i, embed in enumerate(self.order_1_embed)
        ], dim=1)
        order_2 = torch.stack([
            embed(Xf[:, i]) for i, embed in enumerate(self.order_2_embed)
        ], dim=1)

        self.output_order_1 = torch.sum(order_1, dim=1)
        self.output_order_2 = (
            torch.sum(torch.flatten(
                order_2 @ order_2.transpose(1, 2), start_dim=1), dim=1)
            - torch.sum(order_2 * order_2, dim=(1, 2))
        ) / 2
        return order_2, torch.unsqueeze(
            torch.sigmoid(self.output_order_1 + self.output_order_2), dim=1
        )


class DNNUnit(C_Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        # self.norm = torch.nn.BatchNorm1d(output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, x: T_Tensor):
        # return self.activation(self.norm(self.dropout(self.linear(x))))
        return self.activation(self.dropout(self.linear(x)))


class DNN(C_Module):
    def __init__(self, input_dim: int, hidden_dims: T.List[int], output_dim: int = 1):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            DNNUnit(input_dim, hidden_dims[0], 0.1),
            *[
                DNNUnit(hidden_dims[i], hidden_dims[i+1], 0.1)
                for i in range(len(hidden_dims) - 1)
            ],
            DNNUnit(hidden_dims[-1], output_dim, 0.1)
        ])

    def forward(self, x: T_Tensor):
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(x)


class DeepFM(C_Module):
    def __init__(self, embed_dim: int, feature_dims: T.List[int], hidden_dims: T.List[int]):
        super().__init__()
        self.fm = FM(embed_dim, feature_dims)
        self.dnn = DNN(embed_dim * len(feature_dims), hidden_dims)

    def forward(self, Xf: T_Tensor):
        order_2, fm_result = self.fm(Xf)
        dnn_result = self.dnn(torch.flatten(order_2, start_dim=1))
        return torch.sigmoid(fm_result + dnn_result)


class TextModel(C_Module):

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.bert = C_BertModel.from_pretrained('bert-base-uncased')  # type: ignore
        self.gru = torch.nn.GRU(768, hidden_dim, num_layers, batch_first=True)

    def forward(self, X: T_Tensor):
        X = self.bert(X).last_hidden_state
        # _, (X, _) = self.lstm(X)
        _, X = self.gru(X)
        return X[-1, :, :]

class UnifiedModel(C_Module):

    def __init__(self,
                 fm_embed_dim: int, fm_feature_dims: T.List[int],  # FM
                 dnn_hidden_dim: T.List[int], dnn_output_dim: int,  # DNN
                 text_output_dim: int, text_num_layers: int,  # Text
                 output_dim: int,
                 enable_fm: bool = True, enable_dnn: bool = True, enable_text: bool = True
                 ):
        super().__init__()
        self.enable = [enable_fm, enable_dnn, enable_text]
        self.fm = FM(fm_embed_dim, fm_feature_dims)
        self.dnn = DNN(fm_embed_dim * len(fm_feature_dims), dnn_hidden_dim, dnn_output_dim)
        self.text = TextModel(text_output_dim, text_num_layers)
        fc_dim = itertools.compress([1, dnn_output_dim, text_output_dim], self.enable)
        self.fc = torch.nn.Linear(sum(fc_dim), output_dim)

    def forward(self, x: T.Tuple[T_Tensor, T_Tensor]):
        '''
        Xf: Long tensor, shape of (N, field_num)
        Xt: Long tensor, shape of (N, seq_len)
        '''
        Xf, Xt = x
        order_2, fm_result = self.fm(Xf)
        dnn_result = self.dnn(torch.flatten(order_2, start_dim=1))
        text_result = self.text(Xt)
        enable_list = itertools.compress([fm_result, dnn_result, text_result], self.enable)
        result = torch.cat(list(enable_list), dim=1)
        return torch.softmax(self.fc(result), dim=1)

# ------------------------------------------------------------------


def load_data(file_path: str, keys: T.List[str]) -> T.Generator[T.List[T.Any], None, None]:
    with lzma.open(file_path, 'rb') as f:
        for line in f:
            ret = json.loads(line)
            yield [ret[k] for k in keys]

# ---------------------- FEATURE ENGINEERING -----------------------
# Below is the feature engineering part of the preprocessing.
# ------------------------- LDA PROCESSING -------------------------


def LDA_fit(train: T_Sentence):
    dictionary = C_Dictionary(train)
    dictionary.filter_n_most_frequent(200)
    train_corpus = [dictionary.doc2bow(text) for text in train]
    lda = C_LDAModel(corpus=train_corpus, id2word=dictionary, num_topics=8)
    return dictionary, lda


def LDA_predict(lda: C_LDAModel, dictionary: C_Dictionary, sentence: T_Sentence):
    corpus = [dictionary.doc2bow(text) for text in sentence]
    result = lda.get_document_topics(corpus)
    return [max(_, key=lambda x: x[1])[0] for _ in result]  # type: ignore


def build_word_features(lda: C_LDAModel, topic: int, topn: int):
    return lda.show_topic(topic, topn=topn)

# ------------------------ CATEGORY ENCODING -----------------------


def field_encode(data: T_Record, field_name: str) -> T.List[int]:
    return C_LabelEncoder().fit_transform(data[field_name])  # type: ignore


productId_encode = functools.partial(field_encode, field_name='product/productId')
productTitle_encode = functools.partial(
    field_encode, field_name='product/title')
userId_encode = functools.partial(
    field_encode, field_name='review/userId')

# ---------------------- BERT WORD EMBEDDING -----------------------

STATIC_BERT = C_BertModel.from_pretrained('bert-base-uncased')  # type: ignore
STATIC_BERT_TOKENIZER = C_BertTokenizer.from_pretrained(
    'bert-base-chinese')  # type: ignore


def word2vec(word: str) -> T_Tensor:
    return STATIC_BERT(STATIC_BERT_TOKENIZER([word])).last_hidden_state[0][0]


def words2vec(words: T.List[str]) -> T_Tensor:
    return torch.cat([word2vec(word) for word in words], dim=0)


def topic2vec(lda: C_LDAModel, topn: int) -> T.Dict[int, T_Tensor]:
    return {
        topic: words2vec([_[0] for _ in build_word_features(lda, topic, topn)])
        for topic in range(lda.num_topics)
    }

# ----------------------- BUILD DATA BAG ---------------------------


def preprocess_features(data: T_Record, lda: C_LDAModel, dict_: C_Dictionary, topn: int):
    categories_field = LDA_predict(lda, dict_, data['review/summary'])
    referenceId_field = productId_encode(data)
    referenceName_field = productTitle_encode(data)
    productColor_field = userId_encode(data)
    # topic2vecMap = topic2vec(lda, topn)
    # TODO 可靠性？因为这里实际上也相当于对categories_field进行了编码
    return list(zip(*[
        categories_field,
        referenceId_field,
        referenceName_field,
        productColor_field
    ])), [
        max(categories_field) + 1,
        max(referenceId_field) + 1,
        max(referenceName_field) + 1,
        max(productColor_field) + 1
    ]


def preprocess_tokens(sentence: T_Sentence) -> T.List[T_Tensor]:
    tokenizer = C_BertTokenizer.from_pretrained('bert-base-chinese')
    return [
        T_Tensor(tokenizer.convert_tokens_to_ids(line)[:512])
        for line in sentence
    ]

# ------------------------------------------------------------------
# ------------------- TRAIN TEST SPLIT -----------------------------


def multiple_get(data: T.Dict[T_X, T_Y], keys: T.List[T_X]) -> T.List[T_Y]:
    return [data[key] for key in keys]


def transpose(data: T.Sequence[T_Record]) -> T_Record:
    return {k: [d[k] for d in data] for k in data[0].keys()}


def filter_fields(data: T.Iterator[T_Record], keys: T.List[str]) -> T.Generator[T_Record, None, None]:
    for d in data:
        yield {k: d.get(k, 0) for k in keys}


def preprocess_comments(c: T.List[T_Record]) -> T.Dict[str, T.List[T.Any]]:
    comments = transpose(c)
    comments['review/text'] = [
        re.sub(PUNC_RE, '', c).split(' ')
        for c in comments['review/text']
    ]
    comments['review/summary'] = [
        re.sub(PUNC_RE, '', str(c)).split(' ')
        for c in comments['review/summary']
    ]
    comments['tokens'] = preprocess_tokens(comments['review/text'])
    return comments

def replicate(data: T.Iterable[T_Record], criteria: T.Callable[[T_Record], bool], rep: int):
    for _ in data:
        if criteria(_):
            for __ in range(rep):
                yield _
        else:
            yield _

def build_dataset():
    _, tags, summary = zip(*load_data(DATA_PATH, LV1_FIELDS))
    _train, _test = F_Split(list(
        filter_fields(itertools.chain(*_), LV2_FIELDS),
    ), test_size=0.2)
    train, test = map(preprocess_comments, [_train, _test])  # type: ignore
    dictionary, lda = LDA_fit(train['review/summary'])
    train['features'], feature_sizes = preprocess_features(
        train, lda, dictionary, 0)
    test['features'], _ = preprocess_features(test, lda, dictionary, 0)
    keys = ['features', 'tokens', 'review/score']
    return feature_sizes, (
        [{k: v for k, v in zip(keys, values)}
         for values in zip(*multiple_get(train, keys))],
        [{k: v for k, v in zip(keys, values)}
         for values in zip(*multiple_get(test, keys))]
    )


def collate_fn(batch: T.List[T_Record], dev: torch.device) -> T.Any:
    features, tokens, scores = zip(*[_.values() for _ in batch])
    return (
        T_Tensor(features).to(device=dev, dtype=torch.long),
        F_Padding(tokens, batch_first=True, padding_value=0).to(dev, dtype=torch.long)
    ), (T_Tensor(scores) > 3).to(dev, dtype=torch.long)

# -------------------------- TRAINING ------------------------------


def train(model: C_Module, data: C_DataLoader, **params: T.Any):
    model.train()
    optimizer = params['optim'](model.parameters(), lr=params['lr'])
    loss_fn = params['loss'](weight=params['weight'])
    for _ in range(params['epochs']):
        l = 0
        for x, y in tqdm.tqdm(data):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            l += loss.item()
            loss.backward()
            optimizer.step()
        # Backup model
        torch.save(model, 'model-dnn.pt')
        print(f'Epoch {_ + 1}/{params["epochs"]}: {l/len(data)}')
    return model


def test(model: C_Module, data: C_DataLoader) -> T.Dict[int, T_Tensor]:
    model.eval()
    ret = {_: torch.zeros(4) for _ in range(2)}
    for x, y in tqdm.tqdm(data):
        pred = model(x).detach()
        y_pred = torch.argmax(pred, dim=1)
        for _ in range(2):
            # print(((y == _) & (y_pred == _)))
            TP = ((y == _) & (y_pred == _)).to(torch.long).sum().cpu().item()
            FP = ((y != _) & (y_pred == _)).to(torch.long).sum().cpu().item()
            TN = ((y != _) & (y_pred != _)).to(torch.long).sum().cpu().item()
            FN = ((y == _) & (y_pred != _)).to(torch.long).sum().cpu().item()
            ret[_] += torch.tensor([TP, FP, TN, FN])
    return ret

def display_metric(data: T.Dict[int, T_Tensor]):
    for k, v in data.items():
        TP, FP, TN, FN = v
        print(f'Class {k}:')
        print(f'Precision: {TP / (TP + FP)}')
        print(f'Recall: {TP / (TP + FN)}')
        print(f'F1: {2 * TP / (2 * TP + FP + FN)}')
        print()


# ------------------------------------------------------------------

if __name__ == '__main__':
    feature_sizes, dataset = build_dataset()
    collate = functools.partial(collate_fn, dev=DEV)
    DataLoader = functools.partial(
        C_DataLoader, collate_fn=collate, **LOADER_PARAMS)
    train_loader, test_loader = map(DataLoader, dataset)
    model = UnifiedModel(
        64, feature_sizes, [32, 16, 8], 1, 8, 1, 2,
        enable_fm=True, enable_dnn=True, enable_text=False
    ).to(DEV)
    for _ in range(5):
        train(model, train_loader, **TRAIN_PARAMS)
        # model = torch.load('model.pt')
        display_metric(test(model, test_loader))
