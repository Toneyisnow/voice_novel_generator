import torch
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
from TTS.utils.audio.numpy_transforms import save_wav
import torchaudio
from TTS.utils import audio
from scipy.io.wavfile import write
import numpy as np


def retrieve_tts_multi():
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('Using device ', device)
    # List available 🐸TTS models
    print(TTS().list_models())

    # Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return tts

def retrieve_tts_cn():
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Using device ' + device)

    # List available 🐸TTS models
    print(TTS().list_models())

    # Init TTS
    tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=False).to(device)
    return tts


def direct_transcript():

    print("start directing...")

    ### model_name = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    manager = ModelManager()
    model_path, config_path, model_item = manager.download_model(model_name)

    synth = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path
    )


    tts = retrieve_tts_multi()

    raw_text = """
        还没过一分钟，老太太就因为疼痛和筋疲力尽一下子晕了过去。
        两个歹徒从她无力的手指间夺走挎包，逃之夭夭，只留下老太太四仰八叉地躺在人行道上。
        没人目击袭击和抢劫的经过。过了差不多十五分钟后，哈特曼太太被一名路人发现。警察和救护车同时抵达现场，但到了那时，两名歹徒早已逃得没影了。
        哈特曼太太被人用担架送到救护车上，其间她苏醒了一会儿。她转过头，用那对充满痛苦的眼眸望着那名站在她身旁、低头看着她的制服警察。
        """

    # tts.tts_to_file(text=raw_text,
    #                 language="zh-cn",
    #                 speaker_wav="resources/voices/liuchanhg_happy.wav",
    #                 file_path="output_happy.wav")

    synth.tts_to_file(
        text="我的钱，他们抢走了我的皮包，里面放着我的全副身家。",
        language="zh-cn",
        file_path="output_short_angry.wav",
        style_wav="resources/voices/liuchanhg_angry.wav"  # 一个参考情绪音频（中文）
    )

    wav = synth.tts(
        text="我的钱，他们抢走了我的皮包，里面放着我的全副身家。",
        style_wav="resources/voices/liuchanhg_angry.wav"
    )
    ## save_wav(wav, synth.output_sample_rate, "output_angry.wav")

    # wav 是一个 numpy 数组或 PyTorch Tensor
    #torchaudio.save("output_angry.wav", torch.tensor([wav]), synth.output_sample_rate)
    ### save_wav(wav=wav, path="output_angry.wav", sample_rate=synth.output_sample_rate)
    ### audio.save_wav(wav, synth.output_sample_rate, "output_angry.wav")

    # Convert to 16-bit PCM and save
    ###write("output_angry.wav", synth.output_sample_rate, (np.array(wav) * 32767).astype(np.int16))



def test_tts():
    print("This is a test")

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List available 🐸TTS models
    print(TTS().list_models())

    # Init TTS
    tts = retrieve_tts_multi()

    # Text to speech to a file
    raw_text = """
    不幸的是，老太太不是一名精壮男子的对手，更何况有两个精壮男子。还没过一分钟，老太太就因为疼痛和筋疲力尽一下子晕了过去。
    两个歹徒从她无力的手指间夺走挎包，逃之夭夭，只留下老太太四仰八叉地躺在人行道上。
    没人目击袭击和抢劫的经过。过了差不多十五分钟后，哈特曼太太被一名路人发现。警察和救护车同时抵达现场，但到了那时，两名歹徒早已逃得没影了。
    哈特曼太太被人用担架送到救护车上，其间她苏醒了一会儿。她转过头，用那对充满痛苦的眼眸望着那名站在她身旁、低头看着她的制服警察。
    """

    #tts.tts_to_file(text=raw_text, speaker_wav="resources/voices/sample_voice_2.mp3", language="zh-cn", file_path="output_2.wav")

    ###
    tts.tts_to_file(text=raw_text, speaker_wav="resources/voices/sample_voice_2.mp3", language="zh-cn", file_path="output_4.wav")

    pass
