#coding:utf-8

import json

def build_writing_prompt(prompt_data, test_model_sees_criteria=False):
    writing_prompt = "You are a talented creative writer of compelling, original prose.\n\n"
    # judging_criteriaがある場合、必要ならtest_model_sees_criteriaによって表示
    if test_model_sees_criteria and 'judging_criteria' in prompt_data:
        criteria_list = []
        for criteria_set in prompt_data['judging_criteria']:
            criteria_list += criteria_set['criteria']
        writing_prompt += "You are taking a creative writing test. Here are the assessment criteria:\n"
        writing_prompt += "\n".join(criteria_list) + "\n\n"

    main_prompt = prompt_data['writing_prompt']
    if 'seed_modifiers' in prompt_data and prompt_data['seed_modifiers']:
        seed_mod = prompt_data['seed_modifiers'][0] # 1つ目を適用
        main_prompt = main_prompt.replace("<SEED>", seed_mod)
    else:
        main_prompt = main_prompt.replace("<SEED>", "")

    writing_prompt += main_prompt
    return writing_prompt

# 入力データセット
input_file = "creative_writing_prompts_v2.2.json"
output_file = "reconstructed_dataset.json"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 出力用リスト
output_data = []

for prompt_id, prompt_data in data.items():
    # prompt_dataには "writing_prompt", "judging_criteria", "seed_modifiers"等が含まれる
    writing_prompt = build_writing_prompt(prompt_data, test_model_sees_criteria=False)

    # 元データの一部をコピーする例:
    # writing_promptだけでなく、元の "reference_output" や "judging_criteria"なども格納可能
    # 必要なキーを選択して含める
    reconstructed_entry = {
        "id": prompt_id,
        "original_writing_prompt": prompt_data['writing_prompt'],
        "reconstructed_writing_prompt": writing_prompt
    }

    # 必要なら、judging_criteriaなどもコピー
    if 'judging_criteria' in prompt_data:
        reconstructed_entry['judging_criteria'] = prompt_data['judging_criteria']

    # データセットとして使いやすいよう、元データのキーをそのまま組み込み可能
    # 例えば "seed_modifiers"や "reference_output" などのキーがあれば追加:
    for key in ['reference_output', 'seed_modifiers']:
        if key in prompt_data:
            reconstructed_entry[key] = prompt_data[key]

    output_data.append(reconstructed_entry)

# JSONとして出力
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(output_data, out_f, ensure_ascii=False, indent=4)

print("Reconstructed dataset saved to", output_file)
