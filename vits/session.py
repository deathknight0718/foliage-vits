import numpy
import torch
import librosa

from transformers import AutoModelForMaskedLM, AutoTokenizer, TensorType
from modules.models import SynthesizerTraining, SyntheticHubertModel
from io import BytesIO


class NestedObject:

    def __init__(self, input):
        for key, value in input.items():
            if isinstance(value, dict):
                setattr(self, key, NestedObject(value))
            else:
                setattr(self, key, value)


class Session:

    VITS_MODEL_PATH = "/home/foliage/project/foliage-vits/.models/naraka_guqinghan_e15_s180.pth"

    HUBERT_MODEL_PATH = "/home/foliage/project/foliage-vits/.models/chinese-hubert-base"

    ROBERT_MODEL_PATH = "/home/foliage/project/foliage-vits/.models/chinese-roberta-wwm-ext-large"

    REFERENCE_WAV_PATH = "/home/foliage/project/foliage-vits/data/reference.wav"

    DEVICE_MODE_CUDA = "cuda"

    DEVICE_MODE_CPU = "cpu"

    def __init__(self):
        self.bert = ""

    def vits_model(self):
        data = torch.load(Session.VITS_MODEL_PATH, map_location=Session.DEVICE_MODE_CPU)
        self.hyperparameters = NestedObject(data["config"])
        self.hyperparameters.model.semantic_frame_rate = "25hz"
        model = SynthesizerTraining(  #
            self.hyperparameters.data.filter_length // 2 + 1,  #
            self.hyperparameters.train.segment_size // self.hyperparameters.data.hop_length,  #
            n_speakers=self.hyperparameters.data.n_speakers,  #
            **self.hyperparameters.model  #
        )
        model = model.to(Session.DEVICE_MODE_CUDA)
        model.eval()
        model.load_state_dict(data["weight"], strict=False)
        return model

    def hubert(self):
        model = SyntheticHubertModel(Session.HUBERT_MODEL_PATH)
        model.to(Session.DEVICE_MODE_CUDA)
        model.eval()
        return model

    def robert(self):
        model = AutoModelForMaskedLM.from_pretrained(Session.ROBERT_MODEL_PATH)
        model.to(Session.DEVICE_MODE_CUDA)
        return model

    def tokenizer(self):
        model = AutoTokenizer.from_pretrained(Session.ROBERT_MODEL_PATH)
        return model

    def tokens(self, inputText):
        with torch.no_grad():
            tokens = self.tokenizer()(inputText, return_tensors=TensorType.PYTORCH)
            for i in tokens:
                tokens[i] = tokens[i].to(Session.DEVICE_MODE_CUDA)
            result = self.robert()(**tokens, output_hidden_states=True)
            result = torch.cat(result["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(inputText) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = result[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)

    def prompt():
        ""

    def speech(self, text):
        vits_model = self.vits_model()
        wav0 = numpy.zeros(int(self.hyperparameters.data.sampling_rate * 0.3), numpy.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(Session.REFERENCE_WAV_PATH, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(Session.DEVICE_MODE_CUDA)
            wav0 = torch.from_numpy(wav0).to(Session.DEVICE_MODE_CUDA)
            wav16k = torch.cat([wav16k, wav0])
            content = self.hubert().model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = vits_model.extract_latent(content)
            prompt_semantic = codes[0, 0]


if __name__ == "__main__":
    session = Session()
    session.tokens("在语言自然处理领域中，预训练语言模型")
