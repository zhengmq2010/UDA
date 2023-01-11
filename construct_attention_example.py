from itertools import islice
import numpy as np
import csv
from tqdm import tqdm
import json


def load_passages(ctx_file: str):
    docs = {}
    print('Reading data from: %s', ctx_file)
    with open(ctx_file, encoding='utf8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', )
        # file format: doc_id, doc_text, title
        for row in tqdm(reader):
            if row[0] != 'id':
                docs[row[0]] = (row[1], row[2])
    print(f'wiki_psgs loading finished!!')
    return docs


def get_result_dict(file, n_context, save):  # load attention_score.tsv
    result = {}
    # qid::q_text::pid::golden_answer::predict_answer::original_score::attention_score
    with open(file) as f:
        for line in islice(f, 1, None):
            qid, q_txt, pid, golden, pred, score1, score2 = line.strip().split('::')
            qid, q_txt, pid, golden, pred, score1, score2 = qid.strip(':'), \
                                                            q_txt.strip(':'), \
                                                            pid.strip(':'), \
                                                            golden.strip(':'), \
                                                            pred.strip(':'), \
                                                            score1.strip(':'), \
                                                            score2.strip(':')
            if qid not in result.keys():
                result[qid] = {'qid': qid,  # question unique id
                               'q_txt': q_txt,  # question context
                               'gold': golden,  # None
                               'pred': pred,  # None
                               'atn_rank': [],  # re-ranked position by cross-attention score
                               'ctxs': [],  # passages context
                               }

            result[qid]['ctxs'].append({'pid': pid,  # passages unique id
                                        'org_score': score1,  # score from DPR
                                        'atn_score': score2})  # score from cross-attention module
            if len(result[qid]['ctxs']) == n_context:
                attention_score = np.array([float(ctx['atn_score']) * -1 for ctx in result[qid]['ctxs']])
                atn_re_rank = attention_score.argsort()
                result[qid]['atn_rank'] = atn_re_rank
    if save:
        json.dump(result, open(save, 'w'))
    print('reading attention score file finished!!')

    return result


def convert_to_dpr_train(result, save):
    # maintaining same data format with DPR training file
    dpr_train = []
    for qid, p in result.items():
        tmp = {'dataset': 'mrqa_attention_filter',
               'question': p['q_txt'],
               'question_id': p['qid'],
               'answers': [p['pred']],
               'positive_ctxs': [],
               'negative_ctxs': [],
               'hard_negative_ctxs': []}
        for atn_r in p['atn_rank']:
            cur_ctx = p['ctxs'][atn_r]
            if float(cur_ctx['atn_score']) >= 4.5:
                tmp['positive_ctxs'].append({'title': all_passages[cur_ctx['pid']][1],
                                             'text': all_passages[cur_ctx['pid']][0],
                                             'score': 1000,
                                             'atn_score': float(cur_ctx['atn_score']),
                                             'title_score': 1,
                                             'passage_id': cur_ctx['pid'],
                                             }
                                            )
            elif float(cur_ctx['atn_score']) < 3.5:
                tmp['hard_negative_ctxs'].append({'title': all_passages[cur_ctx['pid']][1],
                                                  'text': all_passages[cur_ctx['pid']][0],
                                                  'score': float(cur_ctx['org_score']),
                                                  'atn_score': float(cur_ctx['atn_score']),
                                                  'title_score': 0,
                                                  'passage_id': cur_ctx['pid'],
                                                  }
                                                 )
        if len(tmp['positive_ctxs']) >= 1:
            dpr_train.append(tmp)
    if save:
        json.dump(dpr_train, open(save, 'w'))
    return dpr_train


wiki_path = ''
atn_file = ''
save_dpr_train = ''

all_passages = load_passages(wiki_path)
result = get_result_dict(file=atn_file, n_context=20, save=None)
convert_to_dpr_train(result, save=save_dpr_train)
