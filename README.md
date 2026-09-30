# Conditioning LLMs on Social Value Orientation improves behavioural alignment in a sequential social dilemma

Software for reproducing the results in "Conditioning LLMs on Social Value Orientation improves behavioural alignment in a sequential social dilemma".

The figures can be reproduced using the .ipynb notebooks in notebooks.

The data can be created running src/main.py.
src/main.py takes as parameters:
- model (gpt_5.1 gpt_5.4 gpt_5.4_mini gpt_5.4_nano Llama_3.3_70B_Instruct Qwen3.5_9B Qwen3.5_27B Gemma_4_31B) see data/models.json
- task (cg svo)
- value (any value >= 0, needed only for cg)
- version (integer, 1)
- nsample (integer, 3)
- framing (to perform SVO conditioning)
- blinding (to use blinded prompts)
- comprehension (to perform comprehension test in cg)

project/

|- .env 

|- requirements.txt

|- README.md

|- .gitignore

|- src/

|---- main.py

|---- utilsLiteLLM.py

|---- utilsOutput.py

|- prompts/

|---- cg/

|---- svo/

|---- system/

|- output/

|- data/

|---- models.json

|---- svo_values.csv

|---- human_svo_profiles_beta.csv
