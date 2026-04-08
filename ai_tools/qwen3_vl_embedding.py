import math
import base64
import torch
import torch.nn.functional as F
import unicodedata
import logging

import requests
from PIL import Image
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any, Tuple
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLModel,
    Qwen3VLConfig,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache

logger = logging.getLogger(__name__)

# Constants for configuration
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2


# Define output structure for embeddings
@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


# Define model class to compute embeddings
class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    # Extract image features from model
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules accessible through properties
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    # Forward pass through model with input parameters
    # @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        # Pass inputs through the model
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        # Return the model output
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    max_pixels = max_pixels if max_pixels is not None else IMAGE_MAX_TOKEN_NUM * factor**2
    min_pixels = min_pixels if min_pixels is not None else IMAGE_MIN_TOKEN_NUM * factor**2
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels.")

    ratio = max(height, width) / min(height, width)
    if ratio > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {ratio}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    return pil_image.convert("RGB")


def fetch_image(
    ele: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14
) -> Image.Image:
    image = ele.get("image", ele.get("image_url"))
    if image is None:
        raise ValueError("image or image_url is required.")

    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
    image_obj: Optional[Image.Image] = None

    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str) and image.startswith(("http://", "https://")):
        with requests.get(image, stream=True, timeout=30) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = Image.open(bio).copy()
    elif isinstance(image, str) and image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif isinstance(image, str) and image.startswith("data:image"):
        if "base64," not in image:
            raise ValueError("Only base64 data URI images are supported.")
        _, base64_data = image.split("base64,", 1)
        data = base64.b64decode(base64_data)
        with BytesIO(data) as bio:
            image_obj = Image.open(bio).copy()
    elif isinstance(image, str):
        image_obj = Image.open(image)

    if image_obj is None:
        raise ValueError(
            "Unrecognized image input, supports local path, http/https, data URI and PIL.Image."
        )

    image_rgb = to_rgb(image_obj)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            int(ele["resized_height"]),
            int(ele["resized_width"]),
            factor=patch_factor,
        )
    else:
        width, height = image_rgb.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=ele.get("min_pixels"),
            max_pixels=ele.get("max_pixels"),
        )

    return image_rgb.resize((resized_width, resized_height))


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    image_patch_size: int = 14,
) -> Optional[List[Image.Image]]:
    image_inputs = []
    if conversations and isinstance(conversations[0], dict):
        conversations = [conversations]

    for conversation in conversations:
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for ele in content:
                if (
                    "image" in ele
                    or "image_url" in ele
                    or ele.get("type", "text") in ("image", "image_url")
                ):
                    image_inputs.append(
                        fetch_image(ele, image_patch_size=image_patch_size)
                    )
                elif "video" in ele or ele.get("type") == "video":
                    raise ValueError(
                        "Only text+image inputs are supported; video input is not supported."
                    )

    return image_inputs if image_inputs else None


IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384


# Define embedder class for processing inputs and generating embeddings
class Qwen3VLEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        default_instruction: str = "Represent the user's input.",
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.default_instruction = default_instruction

        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        ).to(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side="right"
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": inputs.get("attention_mask"),
        }

    def format_model_input(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict]:

        # Ensure instruction ends with punctuation
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith(
                "P"
            ):
                instruction = instruction + "."

        # Initialize conversation with system prompts
        content = []
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": instruction or self.default_instruction}
                ],
            },
            {"role": "user", "content": content},
        ]

        # Normalize text input to list
        if text is None:
            texts = []
        elif isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Normalize image input to list
        if image is None:
            images = []
        elif not isinstance(image, list):
            images = [image]
        else:
            images = image

        # Add text or image content to conversation
        if not texts and not images:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        # Process each image
        for img in images:
            image_content = None

            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = (
                    img if img.startswith(("http://", "https://")) else "file://" + img
                )
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")

            # Add image input to content
            if image_content:
                content.append(
                    {
                        "type": "image",
                        "image": image_content,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                )

        # Process each text
        for txt in texts:
            content.append({"type": "text", "text": txt})

        return conversation

    # Preprocess input conversations for model consumption
    def _preprocess_inputs(
        self, conversations: List[List[Dict]]
    ) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            images = process_vision_info(
                conversations,
                image_patch_size=16,
            )
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": "NULL"}]}],
                add_generation_prompt=True,
                tokenize=False,
            )

        inputs = self.processor(
            text=text,
            images=images,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
        )
        return inputs

    # Pool the last hidden state by attention mask for embeddings
    @staticmethod
    def _pooling_last(
        hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    # Process inputs to generate normalized embeddings
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> tuple:
        for ele in inputs:
            unsupported = [
                key for key in ("video", "fps", "max_frames") if ele.get(key) is not None
            ]
            if unsupported:
                raise ValueError(
                    "Qwen3VLEmbedder only supports text+image inputs; "
                    f"unsupported keys: {', '.join(unsupported)}"
                )

        conversations = [
            self.format_model_input(
                text=ele.get("text"),
                image=ele.get("image"),
                instruction=ele.get("instruction"),
            )
            for ele in inputs
        ]

        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {
            k: v.to(self.model.device) for k, v in processed_inputs.items()
        }

        outputs = self.forward(processed_inputs)
        embeddings = self._pooling_last(
            outputs["last_hidden_state"], outputs["attention_mask"]
        )

        # Normalize the embeddings if specified
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
