import torch
import json
from pathlib import Path

from testTTS import direct_transcript
from volc.volc_websocket import volc_generate_voice



def main_generate():
    for chapter_id in range(2, 6):
        generate_chapter(chapter_id)


def generate_chapter(chapter_id: int):

    novel_name = "fairy_tales/309_secret_room_copper_door/"

    # 读取 JSON 文件
    with open('resources/novels/' + novel_name + 'transcript_' + str(chapter_id) + '.txt', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取演员和旁白的配音映射
    voice_mapping = {actor["key"]: actor["voice"] for actor in data["metadata"]["actors"]}
    narrative_voice = data["metadata"]["narrative"]["voice"]

    output_file_folder = "output/" + novel_name + "chapter_" + str(chapter_id) + "/"
    folder_path = Path(output_file_folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    count = 0

    # 遍历 paragraphs
    for paragraph in data["contents"]["paragraphs"]:

        count = count + 1
        # if count < 66:
        #     continue

        # if count > 16:
        #     break

        actor_key = paragraph["actor"]
        emotion = paragraph["emotion"] if hasattr(paragraph, "emotion") else None
        text = paragraph["text"]
        text = text.split("）", 1)[1] if "）" in text else text

        output_file_name = paragraph["id"]

        # 获取对应 voice key
        voice_key = narrative_voice if actor_key == "narrative" else voice_mapping.get(actor_key)

        # 调用生成语音的函数
        print(voice_key, text, output_file_name)
        volc_generate_voice(voice_key, text, output_file_folder, output_file_name, emotion)


def main():
    print("Hello, world!")
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())

    direct_transcript()

if __name__ == "__main__":

    main_generate()