import copy
import copyreg
import os
import json
import pickle
#files = ['Java-C++','C++-Java','Java-C#','C#-Java','Python-C++','C++-Python','Python-C#','C#-Python','C++-C#','C#-C++','C-Javascript','Javascript-C','PHP-Java','Java-PHP']
#files = ['Java-C++']
from preprocess import summary_replace

example = {"Java":None, "C#":None, "C++":None, "C":None, "Python":None, "PHP":None, "Javascript":None, "comment":None}

path = "/home/xxx/CodeTranslation/data/snippet_data/"
postfix = {"Java":"java", "C#":"cs", "C++":"cpp", "C":"c", "Python":"py", "PHP":"php", "Javascript":"js"}

files=os.listdir(path)

dataset = {}



for file in files:
    source, target = file.split('-')
    print(source, target)
    # source, target
    #program test
    map_file = path + source + "-" + target + "/" + "test-" + source + "-map.jsonl"
    source_file = path + source + "-" + target + "/" + "test-" + source + "-" + target + "-tok." + postfix[source]
    target_file = path + source + "-" + target + "/" + "test-" + source + "-" + target + "-tok." + postfix[target]
    text_file = "/home/xxx/CodeTranslation/data/map_data/" + source + "-mapping-tok.jsonl"

    with open(source_file, encoding="utf-8") as f1, open(target_file, encoding="utf-8") as f2, \
            open(map_file, encoding="utf-8") as f3, open(text_file, encoding="utf-8") as f4:
        textdict = {}
        for line in f4.readlines():
            line = json.loads(line)
            if (text_file.find("program") == -1):
                summarystr = summary_replace(line["comment_tokens"])
                # summarystr = line["comment_tokens"]
                textdict[line["idx"]] = summarystr
            else:
                summarystr = summary_replace(line["desc_tokens"])
                # summarystr = line["desc_tokens"]
                textdict[line["idx"].split('-')[0]] = summarystr
        for line1, line2, line3 in zip(f1, f2, f3):
            source_code = str(line1.strip())
            target_code = str(line2.strip())
            idx = str(line3.strip())
            comments = textdict[idx]
            if(idx not in dataset.keys()):
                newexample = copy.deepcopy(example)
                newexample[source]= source_code
                newexample[target]= target_code
                newexample["comment"] = comments
                dataset[idx] = newexample
            else:
                dataset[idx][source]= source_code
                dataset[idx][target]= target_code


with open('/home/xxx/multitranslation/publish/data/test_snippet_label_data.pkl', mode="wb") as out:
    pickle.dump(dataset, out)

print(dataset, len(dataset))
