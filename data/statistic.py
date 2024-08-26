import pickle
def load_dataset(name):
    with open("/home/xxx/multitranslation/publish/data/"+name+"_label_data.pkl", mode="rb") as f:
        a = pickle.load(f)
    return a

postfix = {"Java":"java", "C#":"cs", "C++":"cpp", "C":"c", "Python":"py", "PHP":"php", "     ":"js"}

def sta(examples):
    for lingual in postfix.keys():
        num = 0
        for idx in examples.keys():
            instance = examples[idx]
            source_code = instance[lingual]
            if(source_code != None):
                num += 1
        print(lingual, num)
    print(len(examples))




nam=str(input())
examples = load_dataset(name=nam)
train_features = sta(examples)
