from typing import NamedTuple

import torch
from transformers.models.gemma3 import Gemma3ForConditionalGeneration

from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.text_encoders.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnector,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoderModelBase
from ltx_core.text_encoders.gemma.feature_extractor import (
    FeatureExtractorV1,
    FeatureExtractorV2,
)
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer

# Gemma 3 12B text config constants
_GEMMA_HIDDEN_SIZE = 3840
_GEMMA_NUM_LAYERS = 49  # 48 hidden layers + 1 embedding layer

_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


def _create_feature_extractor(transformer_config: dict) -> torch.nn.Module:
    """Select and create the appropriate feature extractor based on config.
    V1: 19B models — per-segment norm → single aggregate_embed
    V2: 22B models — per-token RMS norm → dual video/audio aggregate embeds
    """
    flat_dim = _GEMMA_HIDDEN_SIZE * _GEMMA_NUM_LAYERS

    overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
    if not overlapping_keys:
        aggregate_embed = torch.nn.Linear(flat_dim, _GEMMA_HIDDEN_SIZE, bias=False)
        return FeatureExtractorV1(aggregate_embed=aggregate_embed, is_av=True)

    video_inner_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
    audio_inner_dim = transformer_config["audio_num_attention_heads"] * transformer_config["audio_attention_head_dim"]
    return FeatureExtractorV2(
        video_aggregate_embed=torch.nn.Linear(flat_dim, video_inner_dim, bias=True),
        embedding_dim=_GEMMA_HIDDEN_SIZE,
        audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_inner_dim, bias=True),
    )


class AVGemmaEncoderOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor
    attention_mask: torch.Tensor


class AVGemmaTextEncoderModel(GemmaTextEncoderModelBase):
    """
    AVGemma Text Encoder Model.
    This class combines the tokenizer, Gemma model, feature extractor from base class and a
    video and audio embeddings connectors to provide a preprocessing for audio-visual pipeline.
    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        embeddings_connector: Embeddings1DConnector,
        audio_embeddings_connector: Embeddings1DConnector,
        tokenizer: LTXVGemmaTokenizer | None = None,
        model: Gemma3ForConditionalGeneration | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            feature_extractor_linear=feature_extractor,
            tokenizer=tokenizer,
            model=model,
            dtype=dtype,
        )
        self.embeddings_connector = embeddings_connector.to(dtype=dtype)
        self.audio_embeddings_connector = audio_embeddings_connector.to(dtype=dtype)

    def forward(self, text: str, padding_side: str = "left") -> AVGemmaEncoderOutput:
        # Get hidden states from Gemma
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Run feature extractor (V1 returns (feats, feats); V2 returns (video, audio))
        video_features, audio_features = self.feature_extractor_linear(
            outputs.hidden_states, attention_mask, padding_side
        )
        if audio_features is None:
            audio_features = video_features

        # Run connectors
        connector_mask = self._convert_to_additive_mask(attention_mask, video_features.dtype)

        encoded, encoded_mask = self.embeddings_connector(video_features, connector_mask)
        binary_mask = (encoded_mask < 0.000001).to(torch.int64)
        binary_mask = binary_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
        encoded = encoded * binary_mask

        encoded_audio, _ = self.audio_embeddings_connector(audio_features, connector_mask)

        return AVGemmaEncoderOutput(encoded, encoded_audio, binary_mask.squeeze(-1))


class AVGemmaTextEncoderModelConfigurator(ModelConfigurator[AVGemmaTextEncoderModel]):
    @classmethod
    def from_config(cls: type["AVGemmaTextEncoderModel"], config: dict) -> "AVGemmaTextEncoderModel":
        transformer_config = config.get("transformer", {})
        feature_extractor = _create_feature_extractor(transformer_config)
        embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        audio_embeddings_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)
        return AVGemmaTextEncoderModel(
            feature_extractor=feature_extractor,
            embeddings_connector=embeddings_connector,
            audio_embeddings_connector=audio_embeddings_connector,
        )


AV_GEMMA_TEXT_ENCODER_KEY_OPS = (
    SDOps("AV_GEMMA_TEXT_ENCODER_KEY_OPS")
    .with_matching(prefix="text_embedding_projection.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("text_embedding_projection.", "feature_extractor_linear.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
)
