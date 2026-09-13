from sigver.data.datasets import load_split, DATASET_NAMES

print("known datasets:", DATASET_NAMES)
print()

names = ["institutional", "cedar", "bhsig260_bengali", "bhsig260_hindi"]
per_writer = {"institutional": 28, "cedar": 48,
              "bhsig260_bengali": 54, "bhsig260_hindi": 54}

for split in ("train", "val", "test"):
    counts, imgs = {}, {}
    for n in names:
        w = sum(1 for v in load_split(n).values() if v == split)
        counts[n] = w
        imgs[n] = w * per_writer[n]
    inst = counts["institutional"]
    pub = sum(v for k, v in counts.items() if k != "institutional")
    inst_i = imgs["institutional"]
    pub_i = sum(v for k, v in imgs.items() if k != "institutional")
    print(f"{split}:")
    for n in names:
        print(f"    {n:20s} {counts[n]:>5} writers  ~{imgs[n]:>7,} images")
    print(f"  public adds {pub}/{inst} writers = {pub/inst:+.1%}")
    print(f"  public adds ~{pub_i:,}/{inst_i:,} images = {pub_i/inst_i:+.1%}")
    print()