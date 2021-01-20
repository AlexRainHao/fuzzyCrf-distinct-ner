import codecs
import requests
from tqdm import tqdm
import json
import sys
import multiprocessing as mp
from multiprocessing import Queue, Process, Pool, cpu_count

class nerClient():
    def __init__(self, ):
        pass
    
    def sent_request(self, element, method="POST", connect_to="http://192.168.100.210:9999/parse"):
        res = requests.request(method, url=connect_to,
                               data=element)
        return res.json()


class annoPredict(nerClient):
    def __init__(self):
        super(annoPredict, self).__init__()
        self.predictionLabel = {"RESIDENT": "LOC"}  # dict. E.X. {parser label: represent label}
        self.replaceLabel = {}  # dict E.X. {"TITLE": path}
        
        self.replaceDicts = {}
        
        self.f = None
        self.Anno = "BIOES"
    
    def register(self, label_cls):
        self.predictionLabel.update(label_cls.predictionLabel)
        self.replaceLabel.update(label_cls.replaceLabel)
    
    def pred_for_single_case(self, case):
        """predict target label for a single case through current NER server"""
        text = ''.join(val[0] for key, val in case.items())
        
        try:
            parser_ner_res = self.sent_request(json.dumps({"q": text}))["entities"]
        except:
            return case
        
        for ner_res in parser_ner_res:
            start = ner_res["start"]
            end = ner_res["end"]
            single_begin_flag = 1 if end - start == 1 else 0
            entity = ner_res["entity"]
            
            _ori_label = sum([0 if case[x][1] == 'O' else 1 for x in range(start, end)])
            
            if entity in self.predictionLabel and _ori_label == 0:
                # write to new label
                
                if self.Anno.lower() == 'bioes':
                    if single_begin_flag:
                        case[start][1] = "S-" + self.predictionLabel.get(entity)
                    
                    else:
                        case[start][1] = "B-" + self.predictionLabel.get(entity)
                        case[end - 1][1] = "E-" + self.predictionLabel.get(entity)
                        
                        for _idx in range(start + 1, end - 1):
                            case[_idx][1] = "M-" + self.predictionLabel.get(entity)
                
                
                elif self.Anno.lower() == 'bio':
                    case[start][1] = "B-" + self.predictionLabel.get(entity)
                    for _idx in range(start + 1, end):
                        case[_idx][1] = "O-" + self.predictionLabel.get(entity)
                
                
                else:
                    raise Exception("No support for assigned format exclude BIOES or BIO")
        
        return case
    
    def load_cases(self, filename):
        """
        load each case for dataset as bio or bioes
        E.X.
            Input: single bio/bios case
            Output: {0: ("今", "O"), 1: ("天", "O"),...,}
        """
        lines = codecs.open(filename, encoding='utf-8').readlines()
        
        curr_case = {}
        total = len(lines)
        pbar = tqdm(total=total, file=sys.stdout)
        start_idx = 0
        
        while lines:
            line = lines.pop(0)
            pbar.update(1)
            start_idx += 1
            
            if (line.strip() == '' and curr_case) or start_idx == total:
                yield curr_case
                curr_case = {}
            
            elif line.strip == '' and not curr_case:
                continue
            
            else:
                _token, _label = line.strip().split(' ')
                curr_case[len(curr_case)] = [_token, _label]
    
    def merge_ori_pred_case(self, ori_case, pred_case):
        """pass
        ori_case: Dict, {0: ("今", "O"), 1: ("天", "O"),...,}
        pred_case: Dict, {0: ("今", "O"), 1: ("天", "O"),...,}
        """
        assert len(ori_case) == len(pred_case)
        
        for key in ori_case:
            if ori_case[key][1] != "O" and pred_case[key][1] == "O":
                pred_case[key] = ori_case[key]
        
        return pred_case
    
    def write_case(self, case):
        for idx, (token, label) in case.items():
            self.f.write(token + ' ' + label + '\n')
        
        self.f.write('\n')
    
    def open_dump_file(self, filename):
        self.f = codecs.open(filename, 'a', encoding='utf-8')
    
    def close_dump_file(self):
        self.f.close()
    
    def pipeline(self, filename="./corpus/demo.char", is_backup=True):
        to_fname = filename
        if is_backup:
            to_fname = filename + ".repred"
        
        self.open_dump_file(to_fname)
        
        for case in self.load_cases(filename):
            
            if not case:
                continue
            
            case = self.pred_for_single_case(case)
            
            self.write_case(case)
        
        self.close_dump_file()
