# Using FreeInference with Inspect AI (Simple Method)

## ✅ The Easy Way: Use OpenAI Provider

Since FreeInference is **OpenAI-compatible**, just use Inspect AI's built-in `openai` provider with a custom `base_url`:

```bash
# Set your API keys
export FREEINFERENCE_API_KEY="your-key-here"
export OPENAI_API_KEY="$FREEINFERENCE_API_KEY"  # Alias for compatibility

# Run with custom base URL
inspect eval inspect/emnlp_react/benchmark.py@emnlp_awards_mcq_task \
    --model openai/llama-3.3-70b-instruct \
    --model-base-url https://api.freeinference.org/v1 \
    --limit 5
```

That's it! No custom provider needed! 🎉

## Why This Works

- FreeInference API: `https://api.freeinference.org/v1/chat/completions`
- OpenAI API: `https://api.openai.com/v1/chat/completions`
- **Same format!** Just change the base URL

## Available Models

Use any FreeInference model:

```bash
# Llama models (free)
--model openai/llama-3.3-70b-instruct
--model openai/llama-4-scout
--model openai/llama-4-maverick

# Gemini (paid)
--model openai/gemini-2.5-flash

# GLM (paid)
--model openai/glm-4.5
```

## Tool Support

✅ **Automatically works!** The OpenAI provider already handles:
- Tool definitions
- Tool calls
- Tool responses
- All OpenAI-compatible features

## Full Example

```bash
#!/bin/bash
# run_freeinference.sh

export OPENAI_API_KEY="hyi-e55cHX6B_qW50ZHvrN6EeMfTTGWPAge_ie_CH4yZzpw"

inspect eval \
    inspect/emnlp_react/benchmark.py@emnlp_awards_mcq_task \
    --model openai/llama-3.3-70b-instruct \
    --model-base-url https://api.freeinference.org/v1 \
    --limit 5

# View results
inspect view
```

## Configuration File (Optional)

Create `.env` file:
```bash
OPENAI_API_KEY=hyi-e55cHX6B_qW50ZHvrN6EeMfTTGWPAge_ie_CH4yZzpw
OPENAI_BASE_URL=https://api.freeinference.org/v1
```

Then just run:
```bash
inspect eval inspect/emnlp_react/benchmark.py@emnlp_awards_mcq_task \
    --model openai/llama-3.3-70b-instruct \
    --limit 5
```

## Advantages Over Custom Provider

✅ **No custom code** - use built-in provider  
✅ **Tool support** - already implemented  
✅ **Maintained** - updates automatically with Inspect AI  
✅ **Simpler** - one command, no wrapper script  
✅ **Standard** - works with all Inspect AI features  

## Migration from Custom Provider

**Old way:**
```bash
python run_inspect_with_freeinference.py eval ... \
    --model freeinference/llama-3.3-70b-instruct
```

**New way:**
```bash
inspect eval ... \
    --model openai/llama-3.3-70b-instruct \
    --model-base-url https://api.freeinference.org/v1
```

Or set `OPENAI_BASE_URL` environment variable and skip the flag!

## Cleanup

The custom provider files are no longer needed:
- `freeinference_provider.py` - can be removed
- `run_inspect_with_freeinference.py` - can be removed

Just use standard `inspect eval` with `--model-base-url`! 🚀

