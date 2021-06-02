import pdb
import sys
import os
import shutil
import gzip
import nltk
import random
import csv
import time
import pickle
import datetime
from dateutil import relativedelta


#random.seed(1234)
random.seed(1470)

overlap_fname = set()
overlap_date = set()
neighbor_date = set()
for fname in os.listdir('ref'):
    overlap_fname.add(fname[:14])
    month = fname[8:14]
    month = datetime.datetime.strptime(month, '%Y%m')
    prev_month = month - relativedelta.relativedelta(months=1)
    next_month = month + relativedelta.relativedelta(months=1)
    overlap_date.add(fname[8:14])
    neighbor_date.add(datetime.datetime.strftime(prev_month, '%Y%m'))
    neighbor_date.add(datetime.datetime.strftime(next_month, '%Y%m'))

sample_size = 200000
samples = []
no_id_overlap_samples = []
no_date_overlap_samples = []
gigaword_dir = '/soe/jmflanig/data/gigaword_5/raw/'
sent_count = 0
last_time = time.time()

prev_sample = pickle.load(open("samples_200k_3.pkl", "rb"))
sent_ids = set()
for sent, news_id, count in prev_sample:
    #if news_id.upper() in overlap_fname:
    #if news_id[8:] in overlap_date:
    if news_id[8:] in overlap_date or news_id[8:] in neighbor_date:
        continue
    samples.append((sent, news_id, count))
    sent_ids.add(count)

print(len(overlap_fname), len(samples))

new_samples = []
sample_size = sample_size - len(samples)

for _, gz in enumerate(os.listdir(gigaword_dir)):
    if gz[-3:] != '.gz':
        continue
    news_id = gz[:-3]
    print(_, gz, news_id, time.time() - last_time)
    gzpath = os.path.join(gigaword_dir, gz)
    fname = gz.replace('.gz', '.txt')
    with gzip.open(gzpath, 'rb') as fin:
        with open(fname, 'wb') as fout:
            shutil.copyfileobj(fin, fout)

    gigasrc = open(fname, 'r').readlines()
    lines = []
    paragraph = ''
    pstart = False
    for lnum, lin in enumerate(gigasrc):
        if lin.strip('\n') == '<P>':
            pstart = True
        elif lin[0] != '<' and lin[-1] != '>' and pstart:
            paragraph += lin.strip('\n').strip()
            paragraph += ' '
        elif lin.strip('\n') == '</P>':
            lines.append(paragraph)
            paragraph = ''
            pstart = False
        elif lin == '</DOC>':
            break

    sents = []
    # sentence tokenization
    for paragraph in lines:
        sents.extend(nltk.sent_tokenize(paragraph.strip()))

    sents = set(sents)

    def reservoir_sampling(samples, sent, news_id):
        if len(samples) < sample_size:
            samples.append((sent, news_id, sent_count))
        else:
            ind = random.randint(0, sent_count)
            if ind < sample_size:
                samples[ind] = (sent, news_id, sent_count)

    for x, sent in enumerate(sents):
        #if gz[:14].upper() not in overlap_fname:
        #if gz[8:14] not in overlap_date:
        if news_id[8:14] not in overlap_date and news_id[8:14] not in neighbor_date:
            if sent_count not in sent_ids:
                reservoir_sampling(new_samples, sent, news_id)
        elif x == 0:
            print(gz)
        #reservoir_sampling(samples, sent, news_id)

        sent_count += 1

    os.remove(fname)

pickle.dump(samples + new_samples, open("samples_no_overlap_range_200k_3.pkl", "wb"))
#pickle.dump(samples, open("samples_200k_3.pkl", "wb"))

