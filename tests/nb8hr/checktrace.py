import sys, collections
rows=[l.split() for l in open(sys.argv[1])]
seen=collections.Counter((s,int(i)) for _,s,i in rows)
steps=collections.Counter(int(st) for st,_,_ in rows)
dups=[k for k,v in seen.items() if v>1]
print("micro-batches", len(rows), "unique", len(seen), "dups", len(dups), "steps", len(steps), "max step", max(steps)+1)
redo=[s for s,c in steps.items() if c!=8]
print("steps run more than once:", sorted(redo)[:20])
# order check: sequence of (shard,idx) in first-seen order should be strictly increasing
order=[]; prev=None
for st,s,i in rows:
    k=(s,int(i))
    if k in order[-1:]: continue
print("first", rows[0], "last", rows[-1])
