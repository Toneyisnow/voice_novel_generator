import torch
import json

from testTTS import direct_transcript
from volc.volc_websocket import volc_generate_voice



def main_generate():
    # 读取 JSON 文件
    with open('resources/novels/murder_on_the_orient_express/stage_6/full_play_script.txt', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取演员和旁白的配音映射
    voice_mapping = {actor["key"]: actor["voice"] for actor in data["metadata"]["actors"]}
    narrative_voice = data["metadata"]["narrative"]["voice"]

    # 遍历 paragraphs
    for paragraph in data["contents"]["paragraphs"]:

        actor_key = paragraph["actor"]
        emotion = paragraph["emotion"] if hasattr(paragraph, "emotion") else None
        text = paragraph["text"]
        text = text.split("）", 1)[1] if "）" in text else text

        output_file_name = paragraph["id"]

        # 获取对应 voice key
        voice_key = narrative_voice if actor_key == "narrative" else voice_mapping.get(actor_key)

        # 调用生成语音的函数
        print(voice_key, text, output_file_name)
        volc_generate_voice(voice_key, text, output_file_name, emotion)


def main():
    print("Hello, world!")
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())

    direct_transcript()

if __name__ == "__main__":
    # volc_generate_voice("zh_female_roumeinvyou_emo_v2_mars_bigtts", "但你不必说谎啊，你难道不知道，为了追回你被枪走的社保津贴，我们会一样勤奋地调查，一样努力地破案？", "roumei_angry", "angry")
    # volc_generate_voice("zh_female_roumeinvyou_emo_v2_mars_bigtts",
    #                     "但你不必说谎啊，你难道不知道，为了追回你被抢走的社保津贴，我们会一样勤奋地调查，一样努力地破案？",
    #                     "roumei_fear", "fear")

    main_generate()