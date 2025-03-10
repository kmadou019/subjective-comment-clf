#!/usr/bin/env python3

import re
import pandas as pd
with open("./comment_glob_train_plus.txt") as f:
    text = f.read()

text = text.split("****")

regex = r"\[.*?]" # match anything between square brackets
comment_chunk = dict()
comment = []
chunks = []
for i in range(len(text)):
    chunk = re.split(regex ,text[i].strip())
    comment.append(''.join(chunk))
    chunks.append(chunk)

comment_chunk = {'comment':comment,'chunks':chunks}
df = pd.DataFrame.from_dict(comment_chunk,)
df.to_csv('comment_chunk.csv',index=False)
