from typing import List, Dict, Any, Tuple, Text
import random
import codecs
import gzip
import pickle

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

def _save_examples(examples: List[CaseExam], filepath: Text):
    """pass"""
    with gzip.open(filepath, "wb") as f:
        f.write(pickle.dumps(examples))

def _load_examples(filepath: Text):
    """pass"""
    with gzip.open(filepath, "rb") as f:
        content = pickle.loads(f.read())
        return content

def convert_lines_to_case(lines):
    """转换bio/bioes数据到case数据格式"""
    examples = []
    curr_exam = CaseExam(text="", labels=[])
    
    for line in lines:
        if not line.strip() and not curr_exam.isEmpty():
            examples.append(curr_exam)
            curr_exam = CaseExam(text="", labels=[])
        
        elif not line.strip():
            curr_exam = CaseExam(text="", labels=[])
        
        else:
            token, label = line.strip().split()
            curr_exam.add_token(token)
            curr_exam.add_label(label)
    
    if not curr_exam.isEmpty():
        examples.append(curr_exam)
        curr_exam = CaseExam(text="", labels=[])
    
    return examples

def load_bio_file(filepath):
    """读 bio bioes 数据"""
    with codecs.open(filepath, encoding="utf-8") as f:
        lines = f.read().splitlines()
    
    return lines

def split_cv_dataset(examples: List[CaseExam], ratio = 0.7):
    """pass"""
    train_length = int(len(examples) * ratio)
    random.shuffle(examples)
    
    return examples[:train_length], examples[train_length:]




