from flask import Flask, render_template, request, send_from_directory
import os
from datetime import datetime
import librosa
import numpy as np
import torch
from pathlib import Path
from synthesizer import audio
from synthesizer.hparams import hparams
from synthesizer.models.tacotron import Tacotron
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import text_to_sequence
from vocoder import inference as vocoder
from encoder import inference as encoder
import soundfile as sf
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
GENERATED_FOLDER = "static/generated"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

class VoiceCloner:
    sample_rate = hparams.sample_rate
    hparams = hparams

    def __init__(self, syn_model_fpath: str, voc_model_fpath: str, enc_model_fpath: str, verbose=True):
        self.syn_model_fpath = Path(syn_model_fpath)
        self.voc_model_fpath = Path(voc_model_fpath)
        self.enc_model_fpath = Path(enc_model_fpath)
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Voice Cloner using device: {self.device}")

        self.synthesizer = self._load_synthesizer()
        self.vocoder = self._load_vocoder()
        self.speaker_encoder = self._load_speaker_encoder()

    def _load_synthesizer(self):
        model = Tacotron(
            embed_dims=hparams.tts_embed_dims,
            num_chars=len(symbols),
            encoder_dims=hparams.tts_encoder_dims,
            decoder_dims=hparams.tts_decoder_dims,
            n_mels=hparams.num_mels,
            fft_bins=hparams.num_mels,
            postnet_dims=hparams.tts_postnet_dims,
            encoder_K=hparams.tts_encoder_K,
            lstm_dims=hparams.tts_lstm_dims,
            postnet_K=hparams.tts_postnet_K,
            num_highways=hparams.tts_num_highways,
            dropout=hparams.tts_dropout,
            stop_threshold=hparams.tts_stop_threshold,
            speaker_embedding_size=hparams.speaker_embedding_size
        ).to(self.device)
        model.load(self.syn_model_fpath)
        model.eval()
        return model

    def _load_vocoder(self):
        vocoder.load_model(self.voc_model_fpath)
        return vocoder

    def _load_speaker_encoder(self):
        encoder.load_model(self.enc_model_fpath)
        return encoder

    def extract_speaker_embedding(self, reference_wav):
        wav, _ = librosa.load(reference_wav, sr=hparams.sample_rate)
        embedding = encoder.embed_utterance(wav)
        return embedding

    def synthesize_voice(self, text, reference_wav_path):
        speaker_embedding = self.extract_speaker_embedding(reference_wav_path)
        text_sequence = text_to_sequence(text.strip(), hparams.tts_cleaner_names)
        text_tensor = torch.tensor(text_sequence).long().unsqueeze(0).to(self.device)
        speaker_tensor = torch.tensor(speaker_embedding).float().unsqueeze(0).to(self.device)
        _, mel_spectrogram, _ = self.synthesizer.generate(text_tensor, speaker_tensor)
        mel_spectrogram = mel_spectrogram.squeeze(0)
        generated_waveform = self.vocoder.infer_waveform(mel_spectrogram.detach().cpu().numpy())
        return generated_waveform

# Load model once
cloner = VoiceCloner(
    syn_model_fpath="E://Real-Time-Voice-Cloning//saved_models//default//synthesizer.pt",
    voc_model_fpath="E://Real-Time-Voice-Cloning//saved_models//default//vocoder.pt",
    enc_model_fpath="E://Real-Time-Voice-Cloning//saved_models//default//encoder.pt"
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        ref_audio = request.files.get("ref_audio") or request.files.get("recorded_audio")

        if not text or not ref_audio:
            return render_template("index.html", error="Please upload or record a reference audio and enter some text.")

        # Save the audio file (uploaded or recorded)
        filename = secure_filename(ref_audio.filename)
        ref_audio_path = os.path.join(UPLOAD_FOLDER, filename)
        ref_audio.save(ref_audio_path)

        try:
            generated_audio = cloner.synthesize_voice(text, ref_audio_path)
            output_filename = f"cloned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            output_path = os.path.join(GENERATED_FOLDER, output_filename)
            sf.write(output_path, generated_audio, samplerate=hparams.sample_rate)

            return render_template("index.html", audio_file=output_filename, success="Audio synthesized successfully!")
        except Exception as e:
            return render_template("index.html", error=f"Error: {e}")

    return render_template("index.html")

@app.route("/generated/<filename>")
def download_file(filename):
    return send_from_directory(GENERATED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)

