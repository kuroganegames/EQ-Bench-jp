import json
from tqdm import tqdm
import anthropic
import re
import sys

def translate_prompt(item, raw_file_path, api_key, model="claude-3-5-sonnet-20241022"):
    """
    Anthropic APIを使用して与えられたJSONスニペットの英語部分を
    日本語へ翻訳する関数です。
    item: 翻訳対象のdict
    raw_file_path: 原文をバックアップするファイルパス
    api_key: AnthropicのAPIキー
    model: 使用するAnthropicモデル（デフォルトは例）
    """

    try:
        item_json_string = json.dumps(item, ensure_ascii=False)
    except Exception as e:
        print(f"Error serializing item to JSON: {e}")
        return None

    # 翻訳指示：JSONスニペット内の英語を日本語へ翻訳
    # emotion名やJSON構造は保持する、等の指示は元のtranslate_openai.pyに倣って作成
    translation_prompt = f"""
以下のJSONスニペットから、プロンプト部分と emotion 名称、ならびに reference_answer, reference_answer_fullscale の値部分にある英語をすべて日本語に翻訳してください。
JSONのキーは翻訳せず、"prompt","reference_answer","reference_answer_fullscale","emotion1","emotion2","emotion3","emotion4","emotion1_score","emotion2_score","emotion3_score","emotion4_score"などのキーはそのままにしてください。

また、emotion名は reference_answer と reference_answer_fullscale で同じ翻訳をしてください。
JSON構造はそのまま保持し、有効なJSONとして返してください。
以下が翻訳対象のJSONです:

{item_json_string}
"""

    # Anthropicクライアント初期化
    anthropic_client = anthropic.Anthropic(api_key=api_key)

    try:
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.5,
            system="You are a professional translator. Translate the given English text in the JSON to Japanese while keeping the JSON structure intact.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": translation_prompt
                        }
                    ]
                }
            ]
        )
        translated_text = message.content[0].text

        # 生のレスポンスをバックアップ
        with open(raw_file_path, 'a', encoding='utf-8') as raw_file:
            raw_file.write(translated_text + "\n\n")

    except Exception as e:
        print(f"An error occurred during Anthropc API call: {e}")
        return None

    # JSONとしてパース可能か試みる
    try:
        translated_item = json.loads(translated_text)
        return translated_item
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from translated text: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def validate_format(translated_data):
    """
    translate_openai.py同様のフォーマット検証例
    必要ならここも調整可能
    """
    required_keys = ["prompt", "reference_answer", "reference_answer_fullscale"]
    emotion_keys = ["emotion1", "emotion2", "emotion3", "emotion4"]
    score_keys = ["emotion1_score", "emotion2_score", "emotion3_score", "emotion4_score"]

    for key, item in translated_data.items():
        if not all(k in item for k in required_keys):
            return False, f"Missing one of the required keys in item {key}."
        for ref_key in ['reference_answer', 'reference_answer_fullscale']:
            ref_section = item[ref_key]
            if not all(emotion_key in ref_section for emotion_key in emotion_keys) or not all(score_key in ref_section for score_key in score_keys):
                return False, f"Missing emotion or score keys in '{ref_key}' section of item {key}."
    return True, "All items conform to the specified format."

def validate_emotion_names_v2(translated_data):
    """
    emotion名がprompt内に2回以上登場するなどのバリデーション
    必要に応じて元通り再現
    """
    for key, item in translated_data.items():
        emotion_names = [item["reference_answer"][f"emotion{i}"] for i in range(1, 5)]
        prompt = item["prompt"]
        for emotion_name in emotion_names:
            if prompt.count(emotion_name) < 2:
                return False, f"Emotion name '{emotion_name}' does not appear at least twice in the prompt for item {key}."

        for i in range(1, 5):
            emotion_key = f"emotion{i}"
            if item["reference_answer"][emotion_key] != item["reference_answer_fullscale"][emotion_key]:
                return False, f"Emotion names do not match between 'reference_answer' and 'reference_answer_fullscale' for item {key}."

    return True, "Emotion names are consistent across items."

def process_file(input_file_path, output_file_path, raw_file_path, api_key, start_index=0):
    # try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data_points = json.load(file)

    output_data = {}

    keys = list(data_points.keys())
    for key in tqdm(keys[start_index:], desc="Progress", unit="item"):
        item = data_points[key]

        translated_item = translate_prompt(item, raw_file_path, api_key)
        if translated_item is None:
            raise ValueError(f"Translation failed for item {key}.")

        # フォーマットバリデーション
        format_valid, format_message = validate_format({key: translated_item})
        if not format_valid:
            raise ValueError(f"Format validation failed for item {key}: {format_message}")

        emotion_names_valid, emotion_names_message = validate_emotion_names_v2({key: translated_item})
        if not emotion_names_valid:
            raise ValueError(f"Emotion names validation failed for item {key}: {emotion_names_message}")

        output_data[key] = translated_item

        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, ensure_ascii=False, indent=4)

    print("Translation and validation completed successfully.")

    #except Exception as e:
    #    print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file_path = 'eq_bench_v2_questions_171.json'  # 入力ファイル
    output_file_path = 'eq_bench_v2_questions_171_ja.json' # 出力ファイル
    raw_file_path = 'eq_bench_v2_questions_171_ja_raw.txt' # 生出力バックアップ
    api_key = "" # 実際のAnthropic APIキーを記入

    start_index = 0
    if len(sys.argv) > 1:
        start_index = int(sys.argv[1])

    process_file(input_file_path, output_file_path, raw_file_path, api_key, start_index)
