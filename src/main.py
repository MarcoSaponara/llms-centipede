import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utilsLiteLLM import *
from utilsOutput import *

from litellm import completion


def main():
    # litellm._turn_on_debug()

    parser = argparse.ArgumentParser(description="test")

    parser.add_argument("--model", type=str, required=True, help="Name of the LLM")
    parser.add_argument("--task", type=str, required=True, help="Name of the task")
    parser.add_argument("--version", type=str, required=True, help="Version of the prompt")
    parser.add_argument("--nsample", type=int, required=True, help="number of samples")
    parser.add_argument("--value", type=int, help="payoffs value at the last node of the CG")
    parser.add_argument("--framing", action="store_true", help="framing flag")
    parser.add_argument("--blinding", action="store_true", help="blinding flag")
    parser.add_argument("--comprehension", action="store_true", help="comprehension test")

    args = parser.parse_args()

    model_id = args.model
    task = args.task
    version = args.version
    nsample = args.nsample
    value = args.value
    framing = args.framing
    blinding = args.blinding
    comprehension = args.comprehension

    if blinding:
        system_type = "blinded"
    else:
        system_type = "unblinded"

    if framing:
        system_type += "-framed"

    cg_instructions_fill = {
    'pl1' : 'A',
    'pl2' :'B',
    'take': 'X',
    'pas':'Y',
    'value': value,
    }

    print(f"Model: {model_id}")
    print(f"Task: {task}")
    print(f"Prompt version: {version}")
    print(f"nb sample: {nsample}")

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")

    models_path = os.path.join(data_dir, "models.json")
    with open(models_path, 'r') as f:
        model_list = json.load(f)

    model_info = model_list[model_id]

    model = model_info['model']

    kwargs = model_info['kwargs']

    # print(model)
    # print(kwargs)

    print(litellm.get_llm_provider(model=model))  # returns (model, custom_llm_provider, dynamic_api_key, api_base)
    print(litellm.get_supported_openai_params(model=model))

    # test = completion(
    #     model=model,
    #     messages=[
    #         {"role": "user",
    #          "content": "Return the sentence 'hello world!'"},
    #     ],
    #     **kwargs
    # )
    # print(test)

    llm = CachedLLM(
        model=model,
        db_path="llm_cache.db",
        memory_cache=True,
    )

    prompts_path = os.path.join(
        project_dir,
        "prompts"
    )

    system_prompt_path = os.path.join(prompts_path, "system", f"{system_type}-system-prompt-ablation.txt")
    with open(system_prompt_path, encoding='utf-8') as f:
        system = f.read()

    instructions_path = os.path.join(prompts_path, task, f"{task}-instructions-v{version}.txt")
    with open(instructions_path, encoding='utf-8') as f:
        instructions = f.read()

    output = {}

    if task == 'svo':
        for item in tqdm(range(1, 7)):
            # print(f"item {item}\n")
            item_path = os.path.join(prompts_path, task, f"{task}-item{item}.txt")
            with open(item_path, encoding='utf-8') as f:
                list_choices = f.read()

            messages = construct_messages(system, instructions + list_choices)
            # print(messages)

            results = generate_responses(llm, messages, nsample, task,
                                         print_time=False, **kwargs)

            output[f"item{item}"] = results

    elif task == 'cg':
        assert value >= 0

        instructions = instructions.format(**cg_instructions_fill)
        # print(instructions)



        if comprehension:

            comprehension_path = os.path.join(prompts_path, task, f"{task}-comprehension.txt")
            with open(comprehension_path, encoding='utf-8') as f:
                comprehension = f.read()

            questions = ["In decision node 1, Player A chooses Y. Then Player B chooses X. How many points does Player B receive?",
                         "In decision node 3, Player A chooses X. How many points does Player A receive?",
                         "In decision node 4, Player B chooses X. How many points does Player A receive?",
                         "In decision node 2, Player B chooses Y. Then Player A chooses X. How many points does Player B receive?",
                         "In decision node 5, Player A chooses Y. Then Player B chooses Y as well. How many points does Player A receive?"
                         ]
            correct_answers = ['60', '90', '40', '30', f'{value}']

            overall_prob = 1.

            for q, question in enumerate(questions):
                comprehension_question = comprehension.format(question=question)
                # print(comprehension_question)

                probs = np.empty((nsample, kwargs['n']), dtype=float)
                for sample_idx in range(nsample):
                    comprehension_answer = llm.complete(
                        messages=construct_messages(system, instructions+comprehension_question),
                        sample_idx=sample_idx,
                        **kwargs,
                    )

                    for idx, c in enumerate(comprehension_answer.choices):
                        content = c.logprobs.content

                        prob = 1.0e-10
                        if content is None:
                            if correct_answers[q] in c.logprobs.top_logprobs[0]:
                                prob = math.exp(c.logprobs.top_logprobs[0][correct_answers[q]])

                        else:
                            if c.message.content != correct_answers[q]:
                                print(f'wrong answer: {c.message.content}, correct is {correct_answers[q]}')

                                top_first_token = {str(top_logprob.token): math.exp(top_logprob.logprob) for top_logprob in content[0].top_logprobs}

                                if correct_answers[q] in top_first_token:
                                    prob = top_first_token[correct_answers[q]]
                                else:
                                    if len(correct_answers[q])==2 and len(content)>1:
                                        top_second_token = {str(top_logprob.token): math.exp(top_logprob.logprob) for
                                                            top_logprob in content[1].top_logprobs}
                                        if correct_answers[q][0] in top_first_token and correct_answers[q][1] in top_second_token:
                                            prob = top_first_token[correct_answers[q][0]]*top_second_token[correct_answers[q][1]]

                            else:
                                first_token = str(content[0].token)

                                if first_token == correct_answers[q]:
                                    prob = math.exp(content[0].logprob)

                                elif len(content)>1:
                                    second_token = str(content[1].token)
                                    if first_token+second_token == correct_answers[q]:
                                        prob = math.exp(content[0].logprob)*math.exp(content[1].logprob)
                                    elif len(content)>2:
                                        third_token = str(content[2].token)
                                        if first_token+second_token+third_token == correct_answers[q]:
                                            prob = math.exp(content[0].logprob) * math.exp(content[1].logprob) * math.exp(content[2].logprob)


                        probs[sample_idx, idx] = prob

                avg_prob = np.mean(probs)
                print(f'Correct answer is selected with avg. probability {np.round(avg_prob,2)}')
                overall_prob *= avg_prob
            print(f'Overall success probability {overall_prob}')




        # for role in tqdm(['A', 'B']):
        #
        #     output_format_path = os.path.join(prompts_path, task, f"{task}-output-format2.txt")
        #     with open(output_format_path, encoding='utf-8') as f:
        #         output_format = f.read()
        #
        #     output_format = output_format.format(value=value, node=node)
        #
        #     messages = construct_messages(system, instructions + output_format)
        #
        #     results = generate_responses(llm, messages, nsample, task, output_format = 2, print_time = True, **kwargs)
        #
        #     output[f"role{role}"] = results

        ### OUTPUT FORMAT: TAKE/PASS

        # output_format_path = os.path.join(prompts_path, task, f"{task}-output-format1.txt")
        # with open(output_format_path, encoding='utf-8') as f:
        #     output_format_text = f.read()

        elif framing:
            # framing_path = os.path.join(prompts_path, task, f"{task}-framing.txt")
            # with open(framing_path, encoding='utf-8') as f:
            #     frame_template = f.read()

            human_svo_profiles_path = os.path.join(data_dir, "human_svo_profiles_beta.csv")
            human_svo_profiles = pd.read_csv(human_svo_profiles_path)
            # print(human_svo_profiles.head())

            svo_values_path = os.path.join(data_dir, "svo_values.csv")
            svo_values = pd.read_csv(svo_values_path)
            # print(svo_values.head())

            for counter, row in tqdm(human_svo_profiles.iterrows()):
                fill = {}
                # for idx in range(1, 7):
                #     tmp_df = svo_values[
                #         (svo_values['question_idx'] == idx) & (svo_values['value'] == row[f'{idx}'])].copy()
                #     fill[f'self{idx}'] = int(tmp_df['self'].values[0])
                #     fill[f'other{idx}'] = int(tmp_df['other'].values[0])

                svo_value = row['svo'].astype(int)
                if svo_value > 57:
                    svo_label = 'altruist'
                elif 22 < svo_value <= 57:
                    svo_label = 'prosocial'
                elif -12 < svo_value <= 22:
                    svo_label = 'individualist'
                else:
                    svo_label = 'competitive'

                fill['svo_value'] = svo_value
                fill['svo_label'] = svo_label

                # if blinding:
                #     frame = frame_template.format(**fill)
                #     system_prompt = system + frame
                # else:
                system_prompt = system.format(**fill)

                if counter < 2:
                    print(system_prompt)

                cg_output = get_cg_output(system_prompt=system_prompt,
                                          instructions_prompt=instructions,
                                          llm=llm,
                                          nsample=nsample,
                                          cg_instructions_fill=cg_instructions_fill,
                                          **kwargs,
                                          )

                svo = int(row['svo'])
                output[f'{svo}'] = cg_output
        else:
            output = get_cg_output(system_prompt=system,
                                   instructions_prompt=instructions,
                                   llm=llm,
                                   nsample=nsample,
                                   cg_instructions_fill=cg_instructions_fill,
                                   print_time=True,
                                   **kwargs,
                                   )

    # cost_info = llm.estimate_cost(
    #     messages,
    #     max_tokens=5,
    #     service_tier="flex",
    # )
    # print(cost_info)

    output_file_name = f"{model_id}-{task}"
    if value is not None:
        output_file_name += f"-{value}"

    output_file_name += f"-v{version}-n{nsample}-{system_type}-seq-explicit-ablation.json"

    output_path = os.path.join(project_dir, "output", output_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print(f"Saved JSON to: {output_path}")


if __name__ == "__main__":
    main()
