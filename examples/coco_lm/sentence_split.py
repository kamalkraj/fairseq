import time
import os
from multiprocessing import Manager, Pool, Queue
import tqdm

# import scispacy
# import spacy

# nlp = spacy.load("en_core_sci_sm")

from nltk.tokenize import sent_tokenize


# listens and writes the text files to data_output_path
def listener(q, filename): 
    count = 0
    file = open(filename, 'w')
    while 1:
        m = q.get()

        if m == 'kill-command-123':
            print(f"Finished writing file {filename}.txt")
            break

        for x in m:
            file.write(x)

        count += 1

        if count % 500 == 0:
            print(f"finished writing {count}")
            file.flush()

        

        

def get_text(texts, q):
    try:
        for text in texts:
            text_to_write = []
            # no_sent = len(doc_processed)
            for sent in sent_tokenize(text):
                # if i != no_sent:
                text_to_write.append(sent + "\n")
                # else:
                #     text_to_write.append(sent + "\n\n")
            text_to_write.append("\n")
            q.put(text_to_write)
        return 1
    except Exception as e:
        print(e)
        print(text)
        return 1

# txt files to be processed are in sample folder
manager = Manager()
q = manager.Queue()
pool = Pool()
watcher = pool.apply_async(listener, (q,"pubmed_data_sent/pubmed.train.txt"))

jobs = []

input_file = "pubmed_data/pubmed.train.txt"
data_reader = open(input_file)

bucket_size = 200
bucket = []
count = 0
for i,text in enumerate(data_reader):
    text = text.strip()
    if text:
        bucket.append(text)
        count += 1
    if len(bucket) % bucket_size == 0:
        job = pool.apply_async(get_text, (bucket, q))
        jobs.append(job)
        bucket = []
        
    if count % 100000 == 0:
        time.sleep(10)

for job in jobs:
    job.get()


q.put('kill-command-123')
pool.close()
pool.join()        
