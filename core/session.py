import numpy
import torch
import librosa
import ffmpeg
import soundfile

from transformers import AutoModelForMaskedLM, AutoTokenizer, TensorType
from bert.models import SynthesizerTraining, SyntheticHubertModel
from vall_e.models.t2s_lightning_module import Text2SemanticLightningModule
from language.analyzers import analyze
from mel_spectrogram import spectrogram_torch
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

    VALL_E_MODEL_PATH = "/home/foliage/project/foliage-vits/.models/naraka_guqinghan_e30.ckpt"

    HUBERT_MODEL_PATH = "/home/foliage/project/foliage-vits/.models/chinese-hubert-base"

    ROBERT_MODEL_PATH = "/home/foliage/project/foliage-vits/.models/chinese-roberta-wwm-ext-large"

    REFERENCE_WAV_PATH = "/home/foliage/project/foliage-vits/data/reference.wav"

    OUTPUT_WAV_PATH = "/home/foliage/project/foliage-vits/.out/output.wav"

    DEVICE_MODE_CUDA = "cuda"

    DEVICE_MODE_CPU = "cpu"

    def __init__(self):
        self.bert = ""

    def vits_model(self):
        data = torch.load(Session.VITS_MODEL_PATH, map_location=Session.DEVICE_MODE_CPU)
        metadata = NestedObject(data["config"])
        metadata.model.semantic_frame_rate = "25hz"
        model = SynthesizerTraining(  #
            metadata.data.filter_length // 2 + 1,  #
            metadata.train.segment_size // metadata.data.hop_length,  #
            n_speakers=metadata.data.n_speakers,  #
            **metadata.model  #
        )
        model = model.to(Session.DEVICE_MODE_CUDA)
        model.eval()
        model.load_state_dict(data["weight"], strict=False)
        return model, metadata

    def hubert(self):
        model = SyntheticHubertModel(Session.HUBERT_MODEL_PATH)
        model.to(Session.DEVICE_MODE_CUDA)
        model.eval()
        return model

    def robert(self):
        model = AutoModelForMaskedLM.from_pretrained(Session.ROBERT_MODEL_PATH)
        model.to(Session.DEVICE_MODE_CUDA)
        return model

    def vall_e(self):
        dict = torch.load(Session.VALL_E_MODEL_PATH, map_location=Session.DEVICE_MODE_CPU)
        metadata = dict["config"]
        model = Text2SemanticLightningModule(metadata, "****", is_train=False)
        model.load_state_dict(dict["weight"])
        model = model.to(Session.DEVICE_MODE_CUDA)
        model.eval()
        return model, metadata

    def tokenizer(self):
        model = AutoTokenizer.from_pretrained(Session.ROBERT_MODEL_PATH)
        return model

    def features(self, text, words):
        with torch.no_grad():
            tokens = self.tokenizer()(text, return_tensors=TensorType.PYTORCH)
            for i in tokens:
                tokens[i] = tokens[i].to(Session.DEVICE_MODE_CUDA)
            result = self.robert()(**tokens, output_hidden_states=True)
            result = torch.cat(result["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(words) == len(text)
        features = []
        for i in range(len(words)):
            features.append(result[i].repeat(words[i], 1))
        features = torch.cat(features, dim=0)
        return features.T

    def analyze(self, text, language="zh"):
        phones, words, normalized_text = analyze(text, language)
        features = self.features(normalized_text, words).to(Session.DEVICE_MODE_CUDA)
        return phones, words, features, normalized_text

    def audio(self, path, metadata):
        out, _ = (ffmpeg.input(path, threads=0).output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=int(metadata.data.sampling_rate)).run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True))
        audio = numpy.frombuffer(out, numpy.float32).flatten()
        audio = torch.FloatTensor(audio).unsqueeze(0)
        return spectrogram_torch(audio, metadata.data.filter_length, metadata.data.sampling_rate, metadata.data.hop_length, metadata.data.win_length, center=False)

    def output(self, path, data, rate):
        data = numpy.frombuffer(data.tobytes(), dtype=numpy.int16)
        soundfile.write(path, data, rate, format='wav')

    def speech(self, text):
        vits_model, vits_metadata = self.vits_model()
        wav0 = numpy.zeros(int(vits_metadata.data.sampling_rate * 0.3), numpy.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(Session.REFERENCE_WAV_PATH, sr=16000)
            wav16k = torch.from_numpy(wav16k).to(Session.DEVICE_MODE_CUDA)
            wav0 = torch.from_numpy(wav0).to(Session.DEVICE_MODE_CUDA)
            wav16k = torch.cat([wav16k, wav0])
            content = self.hubert().model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            prompt_semantic = vits_model.extract_latent(content)[0, 0]
        prompt_phones, prompt_words, prompt_features, prompt_normalized_text = session.analyze("很好，又少了一个敌人，我想它会适合你")
        phones, words, features, normalized_text = session.analyze(text)
        features = torch.cat([prompt_features, features], 1).to(Session.DEVICE_MODE_CUDA).unsqueeze(0)
        phoneme_ids = torch.LongTensor(prompt_phones + phones).to(Session.DEVICE_MODE_CUDA).unsqueeze(0)
        phoneme_len = torch.tensor([phoneme_ids.shape[-1]]).to(Session.DEVICE_MODE_CUDA)
        prompt = prompt_semantic.unsqueeze(0).to(Session.DEVICE_MODE_CUDA)
        with torch.no_grad():
            vall_model, vall_metadata = self.vall_e()
            pred_semantic, index = vall_model.model.infer_panel(phoneme_ids, phoneme_len, prompt, features, top_k=vall_metadata['inference']['top_k'], early_stop_num=50 * vall_metadata["data"]["max_sec"])
        pred_semantic = pred_semantic[:, -index:].unsqueeze(0)
        reference = self.audio(Session.REFERENCE_WAV_PATH, vits_metadata).to(Session.DEVICE_MODE_CUDA)
        audio = vits_model.decode(pred_semantic, torch.LongTensor(phones).to(Session.DEVICE_MODE_CUDA).unsqueeze(0), reference).detach().cpu().numpy()[0, 0]
        audio_options = []
        audio_options.append(audio)
        audio_options.append(wav0)
        audio_options = [option if isinstance(option, numpy.ndarray) else option.cpu().numpy() for option in audio_options]
        self.output(Session.OUTPUT_WAV_PATH, (numpy.concatenate(audio_options, 0) * 32768).astype(numpy.int16), vits_metadata.data.sampling_rate)


if __name__ == "__main__":
    session = Session()
    session.speech("怒发冲冠，凭阑处、潇潇雨歇。抬望眼，仰天长啸，壮怀激烈。三十功名尘与土，八千里路云和月。莫等闲、白了少年头，空悲切。")
