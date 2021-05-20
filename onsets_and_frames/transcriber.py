"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

from math import exp, log
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.transforms import MelSpectrogram

from onsets_and_frames.constants import (
    HOP_LENGTH,
    MAX_MIDI,
    MEL_FMAX,
    MEL_FMIN,
    MIN_MIDI,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from onsets_and_frames.data_classes import MusicAnnotation
from onsets_and_frames.dataset import AudioAndLabels

from .lstm import BiLSTM
from .unet import UNet


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features), nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        model_complexity=48,
        predict_velocity=False,
        feed_velocity_to_onset=False,
        add_unet_model=False,
        min_midi=MIN_MIDI,
        max_midi=MAX_MIDI,
        hop_length=HOP_LENGTH,
        sample_rate=SAMPLE_RATE,
        window_length=WINDOW_LENGTH,
        mel_fmin=MEL_FMIN,
        mel_fmax=MEL_FMAX,
    ):
        self.predict_velocity = predict_velocity
        self.feed_velocity_to_onset = feed_velocity_to_onset
        self.add_unet_model = add_unet_model
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.n_mels = input_features
        self.max_mel_value = log(self.window_length)
        self.min_mel_value = -self.max_mel_value
        self._mel_clamp_value = exp(-log(self.window_length))
        super().__init__()

        model_size = model_complexity * 16

        self.melspectrogram = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.window_length,
            win_length=self.window_length,
            hop_length=self.hop_length,
            power=1.0,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=self.n_mels,
        )

        if self.add_unet_model:
            self.unet = UNet(in_channels=1)

        def sequence_model(input_size: int, output_size: int):
            return BiLSTM(input_size, output_size // 2)

        if self.feed_velocity_to_onset:
            self.onset_stack = nn.Sequential(
                ConvStack(input_features + output_features, model_size),
                sequence_model(model_size, model_size),
                nn.Linear(model_size, output_features),
                nn.Sigmoid(),
            )
        else:
            self.onset_stack = nn.Sequential(
                ConvStack(input_features, model_size),
                sequence_model(model_size, model_size),
                nn.Linear(model_size, output_features),
                nn.Sigmoid(),
            )

        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid(),
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size), nn.Linear(model_size, output_features), nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size), nn.Linear(model_size, output_features), nn.Sigmoid()
        )
        if self.predict_velocity:
            self.velocity_stack = nn.Sequential(
                ConvStack(input_features, model_size), nn.Linear(model_size, output_features)
            )

    def forward(self, mel):
        if self.add_unet_model:
            mel = mel.unsqueeze(1)
            prev_length = mel.shape[2]
            if prev_length % 64 != 0:
                padder = torch.zeros(mel.shape[0], mel.shape[1], 64 - (mel.shape[2] % 64), mel.shape[3]).to(mel.device)
                mel = torch.cat([mel, padder], dim=2)
            mel = self.unet(mel)
            if prev_length % 64 != 0:
                mel = mel[:, :, :prev_length, :]
            mel = mel.squeeze(1)
        if self.predict_velocity:
            velocity_pred = self.velocity_stack(mel)
        else:
            velocity_pred = None
        if self.feed_velocity_to_onset:
            mel_and_velocity = torch.cat([mel, velocity_pred.detach()], dim=-1)
            onset_pred = self.onset_stack(mel_and_velocity)
        else:
            onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def mel(self, wav: torch.tensor) -> torch.tensor:
        mel_output = self.melspectrogram(wav.reshape(-1, wav.shape[-1])[:, :-1]).transpose(-1, -2)
        mel_output = torch.log(torch.clamp(mel_output, min=self._mel_clamp_value))
        return mel_output

    def run_on_batch(self, batch: AudioAndLabels) -> Tuple[MusicAnnotation, Dict[str, any]]:
        audio_label = batch.audio
        onset_label = batch.annotation.onset
        offset_label = batch.annotation.offset
        frame_label = batch.annotation.frame
        velocity_label = batch.annotation.velocity

        mel = self.mel(audio_label)
        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        if self.predict_velocity:
            velocity_pred = velocity_pred.reshape(*velocity_label.shape)
        else:
            velocity_pred = None

        predictions = MusicAnnotation(
            onset=onset_pred.reshape(*onset_label.shape),
            offset=offset_pred.reshape(*offset_label.shape),
            frame=frame_pred.reshape(*frame_label.shape),
            velocity=velocity_pred,
        )

        losses = {
            "loss/onset": F.binary_cross_entropy(predictions.onset, onset_label),
            "loss/offset": F.binary_cross_entropy(predictions.offset, offset_label),
            "loss/frame": F.binary_cross_entropy(predictions.frame, frame_label),
        }

        if self.predict_velocity:
            losses["loss/velocity"] = self.velocity_loss(predictions.velocity, velocity_label, onset_label)

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator
