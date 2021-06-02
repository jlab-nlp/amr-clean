import pdb
import nltk
import os
import numpy as np
import csv
from sacrebleu import sentence_bleu
from rouge import Rouge


# load reference documents
id2doc = {}
for txt in os.listdir('ref'):
    print(txt)
    doc = open(os.path.join('ref', txt)).readlines()
    doc = [line for line in doc]
    id_str = txt[:-4].replace('.', '_')
    id2doc[id_str] = doc

evaluator = Rouge(
    metrics=['rouge-l'],
    max_n=4,
    limit_length=True,
    length_limit=100,
    length_limit_type='words',
    apply_avg=False,
    apply_best=True,
    alpha=0.5, # Default F1_score
    weight_factor=1.2,
    stemming=True)

rows = []
rows2 = []
for txt in os.listdir('test'):
    print(txt)
    doc = open(os.path.join('test', txt)).read().strip('\n')
    doc_id = txt.split('.')[0]
    refs = list(set(id2doc[doc_id]))
    doc_words = set(doc.lower().split())
    overlap_count = np.asarray([len(doc_words & set(ref.lower().split())) for ref in refs])
    bleus = np.asarray([sentence_bleu(doc.lower(), [ref.lower()]).score if overlap_count[x] >= 3 else 0 for x, ref in enumerate(refs)])
    #rouge = np.asarray([evaluator.get_scores(doc.lower(), ref.lower())['rouge-l']['f'] if overlap_count[x] >= 3 else 0 for x, ref in enumerate(refs)])
    ref_count = list(zip(refs, bleus))
    #ref_count = list(zip(refs, overlap_count))
    #ref_count = list(zip(refs, rouge))
    
    ref_count = sorted(ref_count, key=lambda x: x[1])
    
    threshold = max(overlap_count)
    if threshold < 3:
        continue
    for overlap, count in ref_count[-20:]:
        rows.append([txt[:-4], doc, overlap.strip('\n'), count])


    print(len(rows))

    '''
    if len(overlapping) > 1:
        max_ind = np.argmax(bleus)
        #overlap = overlapping[max_ind]
        overlap = refs[max_ind]
        #line_num = np.where(overlap_count >= threshold)[0][max_ind]
        line_num = max_ind
    else:
        overlap = overlapping[0]
        line_num = np.argmax(overlap_count)

    rows.append([txt[:-4], doc, overlap.strip('\n'), line_num])
    '''


with open('overlap.bleu.csv', 'w') as output:
    csv_writer = csv.writer(output)
    for row in rows:
        csv_writer.writerow(row)

def write_parse_file(rows, fname):
    with open(fname, 'w') as output:
        for row in rows:
            if row[2] is not None:
                output.write(row[2])
                output.write('\n')

write_parse_file(rows, 'bleu.parse.txt')

