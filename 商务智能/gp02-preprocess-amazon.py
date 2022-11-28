import pandas as pd
import json
import os
import lzma

os.chdir(os.path.dirname(__file__))

if __name__ == '__main__':
    reviews = pd.read_csv('products_reviews2.csv')
    texts = pd.read_csv('reviews_text2.csv')
    content = pd.merge(reviews, texts, on='index')
    with lzma.open('data-amazon.jsonl.xz', 'wt') as f:
        json.dump({
            'comments': content.to_dict(orient='records'),
            'hotCommentTagStatistics': {},
            'productCommentSummary': {}
        }, f, ensure_ascii=False)
