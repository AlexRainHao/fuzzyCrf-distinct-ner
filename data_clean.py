import codecs
from collections import Counter, defaultdict
import random
from LAC import LAC

tokenizer = LAC(mode = "seg")

class CaseExam():
    def __init__(self, text = "", labels = []):
        self.text = text
        self.labels = labels
        
    def add_token(self, token):
        """pass"""
        self.text += token
        
    def add_label(self, label):
        """pass"""
        self.labels.append(label)
        
    def isEmpty(self):
        """pass"""
        return len(self.text) == 0

def load_bio_file(filepath):
    """读 bio bioes 数据"""
    with codecs.open(filepath, encoding = "utf-8") as f:
        lines = f.read().splitlines()
        
    return lines

def write_bio_file(filepath, lines):
    """存 bio bioes 数据"""
    with codecs.open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + '\n')

def label_count_viewer(lines):
    """数据标签的出现次数"""
    
    label_list = defaultdict(int)
    
    for line in lines:
        if not line.strip():
            continue
            
        _, metion = line.split()
        
        if metion == "O":
            continue
            
        _sc, _lb = metion.split('-', maxsplit = 1)
        if _sc in ["S", "B"]:
            label_list[_lb] += 1
    
    print(Counter(label_list))

def get_sublabel_tokens(lines):
    """各标签下的词集"""
    token_list = defaultdict(set)
    
    last_token = ""
    last_label = None
    for line in lines:
        if not line.strip():
            continue
        
        _tk, metion = line.split()
        
        if metion == "O":
            if last_token and last_label:
                token_list[last_label].add(last_token)
                last_token = ""
                last_label = None
            continue
        
        _sc, _lb = metion.split('-', maxsplit = 1)
        if _sc in ["S", "B"]:
            if last_token:
                token_list[last_label].add(last_token)
            
            last_token = ""
            last_token += _tk
            last_label = _lb
            
        elif _sc in ["I", "E"]:
            last_token += _tk
    
    if last_token and last_label:
        token_list[last_label].add(last_token)
        last_token = ""
        last_label = None
        
    return token_list

def random_sublabel_tokens(res_set, p = 0.5, use_seg = True):
    """以比例筛选词进入distant dictionary
        use_seg = True, 将分词后的最后 uni_seg 和 bi_seg 加入词典，用于处理原token比较长的情况
    """
    res_set = set(res_set)
    res_list = list(res_set)
    if use_seg:
        for tokens in tokenizer.run(res_list):
            # uni
            if tokens and len(tokens[-1]) > 1:
                res_set.add(tokens[-1])
            # bi
            if len(tokens) > 1:
                res_set.add(''.join(tokens[-2:]))
        
    res_set = [x for x in list(res_set) if len(x) > 1]
    return random.choices(res_set, k = int(len(res_set) * p))
    
def fix_consecutive_label(lines, tar_label = "EDU"):
    """连接相邻同类型的实体，如BIIEBIE -> BIIIIIE"""
    for id, line in enumerate(lines[1:]):
        if not line.strip() or not lines[id].strip():
            continue
        tk, mt = line.strip().split()
        l_tk, l_mt = lines[id].strip().split()

        if mt == "O" or l_mt == "O":
            continue

        sc, lb = mt.split('-', maxsplit = 1)
        l_sc, l_lb = l_mt.split('-', maxsplit = 1)

        if l_sc == "E" and sc == "B" and lb == l_lb and lb == tar_label:
            lines[id] = tk + ' ' + "I" + "-" + l_lb
            lines[id + 1] = tk + ' ' + "I" + "-" + lb
            
    return lines

def convert_lines_to_case(lines):
    """转换bio/bioes数据到case数据格式"""
    examples = []
    curr_exam = CaseExam(text = "", labels = [])
    
    for line in lines:
        if not line.strip() and not curr_exam.isEmpty():
            examples.append(curr_exam)
            curr_exam = CaseExam(text = "", labels = [])
        
        elif not line.strip():
            curr_exam = CaseExam(text = "", labels = [])
        
        else:
            token, label = line.strip().split()
            curr_exam.add_token(token)
            curr_exam.add_label(label)
    
    if not curr_exam.isEmpty():
        examples.append(curr_exam)
        curr_exam = CaseExam(text="", labels=[])
        
    return examples

def filter_non_label_lines(examples):
    """去除无标签的单句"""

    filted_examples = list(filter(
        lambda exam: sum(map(lambda x: x != "O", exam.labels)) != 0,
        examples
    ))
    return filted_examples

def random_select_sublabel_tokens(token_dict, use_seg_dict = {}):
    mining_token_dict = {}

    for key, tokens in token_dict.items():
        mining_token_dict[key] = list(set(random_sublabel_tokens(tokens,
                                                                 use_seg = use_seg_dict.get(key, False))))
        
    with codecs.open("./corpus/random_distinct_dict_new.txt", "w", encoding = 'utf-8') as f:
        for label, tokens in mining_token_dict.items():
            for tk in tokens:
                f.write(label + " " + tk + '\n')
                
    for key, tokens in mining_token_dict.items():
        print(key, ", ", len(tokens))
    

if __name__ == '__main__':
    lines = load_bio_file("./corpus/demo.char.repred")
    token_dict = get_sublabel_tokens(lines)

    random_select_sublabel_tokens(token_dict, use_seg_dict = {"LOC": False,
                                                              "TITLE": True,
                                                              "EDU": True})