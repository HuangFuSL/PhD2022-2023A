'''
For the amazon review dataset, learn word vector for each word in the review and find top 5 similar words given a word.

Author: HuangFuSL
ID: 2022311931
Date: 2022-10-23
'''

import csv
import re
import os
import typing as T

import gensim.models

#! Change root directory
os.chdir(os.path.dirname(__file__))

#! Type definitions
T_RECORD = T.Mapping[str, T.Any]
T_DATA = T.Sequence[T_RECORD]
T_TRANSLATION = T.Optional[T.Callable[[T.Any], T.Any]]
T_TRANS_TABLE = T.Mapping[str, T_TRANSLATION]
C_Word2Vec = gensim.models.Word2Vec

#! Constants
DATA_PATH = r'reviews_text2.csv'
PUNC_RE = re.compile(r'[.,!?;:()\[\]{}\-&/*\+=_@><#\^\\|]')
DEFAULT_TRANS_TABLE: T_TRANS_TABLE = {
    '': None,
    'index': None,
    'review/text': lambda _: list(filter(None, re.sub(PUNC_RE, ' ', _).lower().split())),
}

#! Data functions


def translate_data(data: T_DATA, trans_table: T_TRANS_TABLE) -> T_DATA:
    ''' Perform preprocessing based on translation table '''
    ret = []
    for record in data:
        new_record = dict(record)
        for field, trans in trans_table.items():
            if field not in record:
                continue
            elif trans is None:
                new_record.pop(field)
            else:
                new_record[field] = trans(new_record[field])
        ret.append(new_record)
    return ret


def preprocess_data(data: T_DATA, trans_table: T.Optional[T_TRANS_TABLE] = None) -> T_DATA:
    if trans_table is not None:
        return translate_data(data, trans_table)
    return data  # directly return


def from_csv(path: str, trans_table: T.Optional[T_TRANS_TABLE] = None) -> T_DATA:
    ''' Load data from csv file '''
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        data = preprocess_data(list(reader), trans_table)

    return data

def squeeze(data: T_DATA, field: str) -> T.List[T.Any]:
    ''' Squeeze data into a sequence '''
    return [record[field] for record in data]

if __name__ == '__main__':
    data = from_csv(DATA_PATH, DEFAULT_TRANS_TABLE)
    model = C_Word2Vec(squeeze(data, 'review/text'), vector_size=32, min_count=1)
    word = input('Input a word: ')
    print(f'Top 5 similar words for {word}:')
    for i, (similar_word, similarity) in enumerate(model.wv.most_similar(word, topn=5), 1):
        print(f'{i} - {similar_word}: {similarity}')
