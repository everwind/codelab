#coding=utf-8
import sys
import re
for line in sys.stdin:
  line = line.strip().decode('utf-8')
  items = line.split('\t')
  #sentence, lable = line.split('\t')
  sentence = items[0]
  ss = [s.encode('utf-8') for s in sentence ]
  ss2 = []
  tmp = ""
  for s in ss:
    if s.isalnum():
      tmp+=s
    else:
      if len(tmp)>0: ss2.append(tmp)
      tmp = ""
      ss2.append(s)
  print " ".join(ss2)
  #for s in sentence:
  #  print(s.encode('utf-8'))
