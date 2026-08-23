uv run vlm-demo -i ./vids -p "Describe the main action briefly in 2~6 words." \
    -m THChou1220/gemma-4-e2b-kinetics54K-enhanced-fall_FFT \
    --pace complete --host 0.0.0.0 --window-sec 3.0 --pass-gap 1.5 \
    --highlight-regex "(?i)fall(s|ing|en)?|slip(s|ped|ping)?|struggl(e|es|ing)"
