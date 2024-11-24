quantization="${1:-q0f32}"
device="${2:-vulkan}"
model=$3
conv_template=$4

python -m mlc_llm convert_weight --model-type llama "./$model/" --quantization "$quantization" --output "./output-$model-$quantization"
python -m mlc_llm gen_config --model-type llama "./$model" --quantization "$quantization" --conv-template "conv_template" --output "./output-$model-$quantization"
python -m mlc_llm compile --model-type llama "./output-$model-$quantization/" --quantization "$quantization" --device "$device" --output "./$model-$quantization.so"
