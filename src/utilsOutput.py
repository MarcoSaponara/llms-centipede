import time

import litellm
import math

from utilsLiteLLM import CachedLLM


# def format_logprobs(response):
#     results = {}
#     for idx, choice in enumerate(response.choices):
#         content = choice.logprobs.content
#         rows = []
#         for pos, token in enumerate(content):
#             row = {
#                 "token": repr(token.token),
#                 "prob": f"{math.exp(token.logprob):.4%}",
#                 "top": [
#                     f"{repr(t.token)}: {math.exp(t.logprob):.4%}"
#                     for t in token.top_logprobs
#                 ],
#             }
#             rows.append(row)
#         results[f'sample-{idx}'] = rows
#     return results
#
#
# def print_logprobs(response):
#     logprobs = format_logprobs(response)
#     for sample, rows in logprobs.items():
#         print(f'\n{sample}')
#         for pos, row in enumerate(rows):
#             print(f"\nToken {pos}: {row['token']}  (p={row['prob']})")
#             print("  Alternatives:")
#             for alt in row["top"]:
#                 print(f"    {alt}")


def get_probs_task(response,
                   task: str,
                   # output_format: int = 1,
                   ):
    assert task in ['svo', 'cg']

    choices = []
    if task == 'svo':
        choices = [str(i) for i in range(1, 10)]
    elif task == 'cg':
        # if output_format == 1:
        #     choices = ['X', 'Y']
        # elif output_format == 2:
        #     choices = [str(i) for i in range(4)]
        choices = ['X', 'Y']

    probs = {}
    for idx, c in enumerate(response.choices):
        # print(c)
        if not hasattr(c, "logprobs"):
            raise RuntimeError(
                "The upstream provider did not return logprobs. "
                "Check LiteLLM debug output and the raw provider response."
            )
        if c.message.content in choices:
            # print(f'Valid response. Response: {c.message.content}')
            d = dict.fromkeys(choices)
            lp = c.logprobs
            if lp.content is None:  # this case is used for huggingface/together/meta-llama/Llama-3.3-70B-Instruct
                first_token = str(lp.tokens[0])
                if first_token in choices:
                    for t, value in lp.top_logprobs[0].items():
                        if str(t) in choices:
                            d[str(t)] = math.exp(value)
            else:
                first_token = str(lp.content[0].token)
                if first_token in choices:
                    for t in lp.content[0].top_logprobs:
                        if str(t.token) in choices:
                            d[str(t.token)] = math.exp(t.logprob)

            probs[f'sample-{idx}'] = d
        else:
            print(f'Invalid response. Response: {c.message.content}')

    return probs


def generate_responses(llm: CachedLLM,
                       messages: list[dict],
                       nsample: int,
                       task: str,
                       # output_format: int = 1,
                       print_time: bool = False,
                       **kwargs,
                       ) -> litellm.ModelResponse:
    results = []
    t0 = time.perf_counter()

    for i in range(nsample):
        response = llm.complete(
            messages=messages,
            sample_idx=i,  # whatever integer, just needs to be different from the previous call to miss the cache
            **kwargs,
        )
        # print(response)

        results.append(get_probs_task(response, task)) #, output_format))

    if print_time:
        t1 = time.perf_counter()
        print(f"time {(t1 - t0) * 1000:.0f} ms")

    return results


def construct_messages(system_prompt: str,
                       user_prompt: str,
                       ) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def construct_output_format(cg_instructions_fill: dict,
                            node : str):
    pl1 = cg_instructions_fill['pl1']
    pl2 = cg_instructions_fill['pl2']
    take = cg_instructions_fill['take']
    pas = cg_instructions_fill['pas']

    if node in ['1', '3', '5']:
        role = pl1
    elif node in ['2', '4', '6']:
        role = pl2

    out_format = f"You are Player {role}."
    if node != '1':
        for nn in range(int(node) - 1):
            subject = 'you'
            if nn % 2 == 0:
                if role == pl2:
                    subject = f'Player {pl1}'
            else:
                if role == pl1:
                    subject = f'Player {pl2}'

            out_format += f" At decision node {nn + 1} {subject} chose {pas}."

    out_format += f" You are at decision node {node}. Return only your choice ({take} or {pas})."
    return out_format

def get_cg_output(system_prompt: str,
                  instructions_prompt: str,
                  llm: CachedLLM,
                  nsample: int,
                  cg_instructions_fill: dict,
                  print_time: bool = False,
                  **kwargs,
                  ):
    pl1 = cg_instructions_fill['pl1']
    pl2 = cg_instructions_fill['pl2']
    # take = cg_instructions_fill['take']
    # pas = cg_instructions_fill['pas']

    role_and_nodes = {pl1: ['1', '3', '5'], pl2: ['2', '4', '6']}
    cg_output = {}
    for role, nodes in role_and_nodes.items():
        tmp = {}
        for node in nodes:

            out_format = construct_output_format(cg_instructions_fill, node)
            # print(out_format)

            messages = construct_messages(system_prompt, instructions_prompt + out_format)  # !!

            results = generate_responses(llm, messages, nsample,
                                         task='cg',
                                         # output_format=1,
                                         print_time=print_time,
                                         **kwargs)
            tmp[node] = results

        cg_output[f"role{role}"] = tmp
    return cg_output
