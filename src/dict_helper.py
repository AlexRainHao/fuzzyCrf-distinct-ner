from typing import List, Text, Dict, Any, Tuple
import codecs
import os.path as opt
import re
from src.data_helper import load_rowline_file, _load_examples, CaseExam

def sort_distinct_dict(fpath):
    """pass"""

    new_dict = []

    with codecs.open(fpath, encoding='utf-8') as f:
        lines = f.read().splitlines()

    for line in lines:
        tokens = line.split(' ')
        metion = tokens.pop(0)

        if len(tokens) == 1:
            new_dict.append(line + '\n')

        else:
            tokens = sorted(tokens, key = lambda x: len(x), reverse = False)

            new_dict.append(metion + ' ' + ' '.join(tokens) + '\n')

    with codecs.open(fpath, "w", encoding='utf-8') as f:
        f.writelines(new_dict)

class dictRow:
    def __init__(self, name: Text, tokens: List[Text]):
        self.name = name
        self.tokens = tokens

    def serialize(self):
        return ' '.join([self.name] + self.tokens)

class dictRefinement:
    def __init__(self, fpathstr: List[Text] = None):
        self.dict_rows = self._load_dict(fpathstr)

    def _load_dict(self, fpathstr: List[Text]) -> List[dictRow]:
        """pass"""
        dict_rows = []

        for fpath in fpathstr:
            if not opt.exists(fpath):
                print(f"not fount {fpath}")
                continue

            lines = load_rowline_file(fpath)
            for line in lines:
                if not line.strip():
                    continue

                tokens = line.split(' ')
                name = tokens.pop(0)

                dict_rows.append(dictRow(name = name, tokens = tokens))

        print(f"load {len(dict_rows)} distant rows")
        return dict_rows

    def _save_dict(self, fpath: Text):
        """pass"""
        with codecs.open(fpath, "w", encoding='utf-8') as f:
            [f.write(row.serialize() + '\n') for row in self.dict_rows]

    def tailoring(self, case_examples: List[CaseExam]) -> List[dictRow]:
        """pass"""
        remains_rows = []

        case_examples = list(filter(lambda case: not case.isEmpty(), case_examples))
        if not self.dict_rows or not case_examples:
            return []

        full_string = ".".join(case.text for case in case_examples)
        for index, row in enumerate(self.dict_rows):
            filter_flag = len(list(filter(lambda token: token in full_string, row.tokens)))

            if filter_flag:
                remains_rows.append(row)

        print(f"remain {len(remains_rows) / len(self.dict_rows):.3f} distinct rows")

        self.dict_rows = remains_rows



if __name__ == '__main__':
    op = dictRefinement(["../data/distinct_dict.txt"])
    examples = _load_examples("../data/train_exams.gz")

    res = op.tailoring(examples)

    op._save_dict("../data/yyh.txt")