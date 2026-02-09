cd evaluation
export OPENAI_API_KEY=""
export OPENAI_API_BASE="https://api.openai.com/v1"

python -m RedCode_Exec.main RA \
  --model gpt-4o-2024-05-13 \
  --temperature 0 \
  --max_tokens 1024 \
  --python_eval