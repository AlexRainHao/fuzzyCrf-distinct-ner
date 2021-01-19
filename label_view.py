import codecs
from collections import Counter


def label_count_viewer(filepath):
    """pass"""
    
    label_list = []
    
    with codecs.open(filepath, encoding = "utf-8") as f:
        lines = f.read().splitlines()
        
    for line in lines:
        if not line.strip():
            continue
            
        _, _lb = line.split()
        label_list.append(_lb.split('-', maxsplit = 1)[-1])
    
    return Counter(label_list)

if __name__ == '__main__':
    res = label_count_viewer("./corpus/demo.train.char")
    print(res)

    

