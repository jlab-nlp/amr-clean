import pdb
import sys
import os
import shutil
import gzip
import nltk


input_file = sys.argv[1]
gigaword_dir = '/soe/jmflanig/data/gigaword_5/raw/'
copied = {}
count = 0
for line in open(input_file).readlines():
    if line[:6] == '# ::id':
        count += 1
        doc_id = line.split()[2][6:20].lower()
        test_id = line.split()[2][6:]
        giga_doc_id = line.split()[2][6:27]
        giga_doc_id = giga_doc_id[:-5] + '.' + giga_doc_id[-4:]
        print(count, doc_id, test_id, giga_doc_id)
        if giga_doc_id in copied:
            continue
        fname = '{}.txt'.format(doc_id)
        gzname = '{}.gz'.format(doc_id)
        gzpath = os.path.join(gigaword_dir, gzname)
        if not os.path.exists(fname):
            with gzip.open(gzpath, 'rb') as fin:
                with open(fname, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)

        gigasrc = open(fname, 'r').readlines()
        giga_doc_id_str = 'DOC id="{}"'.format(giga_doc_id)
        lines = []
        paragraph = ''
        start = 0
        for lnum, lin in enumerate(gigasrc):
            if giga_doc_id_str in lin:
                start = lnum
            elif start > 0 and lnum > start + 3 and lin[0] != '<' and lin[-1] != '>':
                paragraph += lin.strip('\n').strip()
                paragraph += ' '
            elif start > 0 and lnum > start + 3 and lin.strip('\n') == '</P>':
                lines.append(paragraph)
                paragraph = ''
            elif start > 0 and lin == '</DOC>':
                break

        sents = []
        # sentence tokenization
        for paragraph in lines:
            sents.extend(nltk.sent_tokenize(paragraph.strip()))
            
        with open('ref/{}.txt'.format(giga_doc_id), 'w') as output:
            for sent in sents:
                output.write(sent)
                output.write('\n')

        copied[giga_doc_id] = True
        os.remove(fname)
    elif line[:7] == '# ::snt':
        sent = line[8:]
        with open('test/{}.txt'.format(test_id), 'w') as output:
            output.write(sent)

