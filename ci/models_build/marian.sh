quantization="$1"
language_pair="${2:-en-it}"


mlc_llm convert_weight --model-type marian ./opus-mt-$language_pair/ --quantization "$quantization" --output "./output-$quantization-opus-mt-$language_pair"
mlc_llm gen_config --model-type marian ./opus-mt-$language_pair --quantization "$quantization" --conv-template marian --output "./output-$quantization-opus-mt-$language_pair"
mlc_llm compile --model-type marian "./output-$quantization-opus-mt-$language_pair/" --quantization "$quantization" --device vulkan --output "./output-$quantization-opus-mt-$language_pair/opus-mt-$language_pair.so"
