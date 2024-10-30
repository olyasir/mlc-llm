quantization="$1"
mlc_llm convert_weight --model-type marian ./opus-mt-en-it/ --quantization "$quantization" --output "./output-$quantization-opus-mt-en-it"
mlc_llm gen_config --model-type marian ./opus-mt-en-it --quantization "$quantization" --conv-template marian --output "./output-$quantization-opus-mt-en-it"
mlc_llm compile --model-type marian "./output-$quantization-opus-mt-en-it/" --quantization "$quantization" --device vulkan --output "./output-$quantization-opus-mt-en-it/opus-mt-en-it.so"