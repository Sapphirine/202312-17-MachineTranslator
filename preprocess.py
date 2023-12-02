import json
english_train_file = 'data/nejm.train.en'
chinese_train_file = 'data/nejm.train.zh'
english_eval_file = 'data/nejm.dev.en'
chinese_eval_file = 'data/nejm.dev.zh'
english_test_file = 'data/nejm.test.en'
chinese_test_file = 'data/nejm.test.zh'
def merge_to_json(english_file, chinese_file, output_file, train):
    with open(english_file, 'r', encoding='utf-8') as file1, \
         open(chinese_file, 'r', encoding='utf-8') as file2:
        
        english_lines = file1.readlines()
        chinese_lines = file2.readlines()

        # Ensure both files have the same number of lines
        if len(english_lines) != len(chinese_lines):
            raise ValueError("Files do not have the same number of lines")

        translations = []

        for en, zh in zip(english_lines, chinese_lines):
            translation_pair = {"translation": {"en": en.replace("@-@","").strip(), "zh": zh.replace(" ","").strip()}}
            translations.append(translation_pair)
        if  train:
            n = len(translations) - len(translations) % 10000
            translations = translations[:n]
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(translations, outfile, ensure_ascii=False, indent=4)

# Replace 'english.txt' and 'chinese.txt' with your actual file paths
merge_to_json(english_train_file, chinese_train_file, 'translations.json',True)
merge_to_json(english_eval_file, chinese_eval_file, 'validation.json',False)
merge_to_json(english_test_file, chinese_test_file, 'test.json',False)